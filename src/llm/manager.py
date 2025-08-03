"""
LLM manager for handling multiple LLM providers.
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from .base import BaseLLMProvider
from .huggingface_llm import HuggingFaceLLMProvider
from .ollama_llm import OllamaLLMProvider


class LLMManager:
    """Manages multiple LLM providers with automatic selection."""
    
    def __init__(self, default_provider: str = "auto"):
        """
        Initialize the LLM manager.
        
        Args:
            default_provider: Default provider ('huggingface', 'ollama', 'auto')
        """
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.default_provider = default_provider
        self.current_provider: Optional[str] = None
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers based on environment."""
        # Ollama (check if enabled)
        if os.getenv("ENABLE_OLLAMA_LLM", "true").lower() == "true":
            try:
                model = os.getenv("OLLAMA_LLM_MODEL", "mistral:instruct")
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self.providers["ollama"] = OllamaLLMProvider(
                    model_name=model,
                    base_url=base_url
                )
                print(f"Initialized Ollama LLM with model: {model}")
            except Exception as e:
                print(f"Failed to initialize Ollama LLM: {e}")
        
        # HuggingFace
        if os.getenv("ENABLE_HUGGINGFACE_LLM", "true").lower() == "true":
            try:
                model = os.getenv("HUGGINGFACE_LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
                use_api = os.getenv("HUGGINGFACE_LLM_USE_API", "false").lower() == "true"
                device = os.getenv("HUGGINGFACE_LLM_DEVICE", "cpu")
                load_in_4bit = os.getenv("HUGGINGFACE_LLM_4BIT", "false").lower() == "true"
                load_in_8bit = os.getenv("HUGGINGFACE_LLM_8BIT", "false").lower() == "true"
                
                self.providers["huggingface"] = HuggingFaceLLMProvider(
                    model_name=model,
                    use_api=use_api,
                    device=device,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit
                )
                print(f"Initialized HuggingFace LLM with model: {model}")
            except Exception as e:
                print(f"Failed to initialize HuggingFace LLM: {e}")
        
        if not self.providers:
            raise ValueError("No LLM providers could be initialized.")
        
        # Set current provider
        if self.default_provider == "auto":
            self.current_provider = self._select_best_provider()
        elif self.default_provider in self.providers:
            self.current_provider = self.default_provider
        else:
            self.current_provider = list(self.providers.keys())[0]
        
        print(f"Using {self.current_provider} as the default LLM provider")
    
    def _select_best_provider(self) -> str:
        """Select the best provider based on availability and performance."""
        # Priority order
        priority_order = ["ollama", "huggingface"]
        
        for provider in priority_order:
            if provider in self.providers:
                return provider
        
        return list(self.providers.keys())[0]
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None
    ) -> str:
        """
        Generate text using the specified or current provider.
        
        Args:
            prompt: The prompt text
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            provider: Specific provider to use
            
        Returns:
            Generated text
        """
        provider_name = provider or self.current_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        start_time = datetime.now()
        try:
            result = await self.providers[provider_name].generate(
                prompt, system_prompt, temperature, max_tokens
            )
            
            # Record success
            self._record_performance(provider_name, True, (datetime.now() - start_time).total_seconds())
            return result
            
        except Exception as e:
            # Record failure
            self._record_performance(provider_name, False, (datetime.now() - start_time).total_seconds())
            print(f"Error with {provider_name} provider: {e}")
            
            # Try fallback
            for fallback in self.providers:
                if fallback != provider_name:
                    try:
                        print(f"Trying fallback provider: {fallback}")
                        return await self.providers[fallback].generate(
                            prompt, system_prompt, temperature, max_tokens
                        )
                    except Exception as fallback_error:
                        print(f"Fallback {fallback} also failed: {fallback_error}")
            
            return ""
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None
    ) -> str:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            provider: Specific provider to use
            
        Returns:
            Generated response
        """
        provider_name = provider or self.current_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        return await self.providers[provider_name].chat_completion(
            messages, temperature, max_tokens
        )
    
    def _record_performance(self, provider: str, success: bool, latency: float):
        """Record performance metrics."""
        if provider not in self.performance_history:
            self.performance_history[provider] = []
        
        self.performance_history[provider].append({
            "timestamp": datetime.now(),
            "success": success,
            "latency": latency
        })
        
        # Keep only last 100 records
        if len(self.performance_history[provider]) > 100:
            self.performance_history[provider] = self.performance_history[provider][-100:]
    
    def get_provider_info(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a provider."""
        provider_name = provider or self.current_provider
        if provider_name in self.providers:
            return self.providers[provider_name].get_metadata()
        return {}
    
    def list_providers(self) -> List[str]:
        """List all available providers."""
        return list(self.providers.keys())
    
    def switch_provider(self, provider: str):
        """Switch to a different LLM provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not available")
        self.current_provider = provider
        print(f"Switched to {provider} LLM provider")