"""
Ollama LLM provider for local open-source models.
"""
import os
import json
import requests
import asyncio
from typing import List, Dict, Any, Optional
from .base import BaseLLMProvider


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama provider for local LLM inference."""
    
    def __init__(
        self,
        model_name: str = "mistral:instruct",
        base_url: Optional[str] = None,
        max_tokens: int = 4096
    ):
        """
        Initialize Ollama LLM provider.
        
        Args:
            model_name: Ollama model name
            base_url: Ollama API base URL
            max_tokens: Maximum tokens for generation
        """
        super().__init__(model_name, max_tokens)
        
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_endpoint = f"{self.base_url}/api/generate"
        self.chat_endpoint = f"{self.base_url}/api/chat"
        
        # Test connection
        self._test_connection()
        
        # Set metadata
        self.metadata = {
            "is_local": True,
            "base_url": self.base_url
        }
    
    def _test_connection(self):
        """Test connection to Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running (ollama serve)."
            )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Ollama."""
        max_tokens = max_tokens or self.max_tokens
        
        # Format with system prompt if provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            # Run in executor to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=120  # Long timeout for generation
                )
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            print(f"Error generating with Ollama: {e}")
            return ""
    
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """Generate text for multiple prompts."""
        # Ollama doesn't support batch processing, so process sequentially
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, system_prompt, temperature, max_tokens)
            results.append(result)
        return results
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a chat completion using Ollama's chat endpoint."""
        max_tokens = max_tokens or self.max_tokens
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    self.chat_endpoint,
                    json=payload,
                    timeout=120
                )
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                print(f"Ollama chat API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            print(f"Error with Ollama chat: {e}")
            return ""
    
    def get_cost_per_token(self) -> Dict[str, float]:
        """Get cost per token (free for local Ollama)."""
        return {
            "input": 0.0,
            "output": 0.0
        }
    
    async def pull_model(self, show_progress: bool = True) -> bool:
        """Pull model from Ollama registry if not available."""
        try:
            # Check if model exists
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Model {self.model_name} is already available")
                return True
            
            # Pull the model
            print(f"Pulling model {self.model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                stream=show_progress,
                timeout=None
            )
            
            if show_progress:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        print(f"\r{status}", end="", flush=True)
                print()
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return [model["name"] for model in models]
                
        except Exception as e:
            print(f"Error listing models: {e}")
        
        return []
    
    @staticmethod
    def get_recommended_models() -> Dict[str, Dict[str, Any]]:
        """Get list of recommended Ollama models."""
        return {
            "mistral:instruct": {
                "size": "7B",
                "description": "Mistral 7B Instruct - Fast and efficient",
                "pull_command": "ollama pull mistral:instruct"
            },
            "llama2:7b-chat": {
                "size": "7B", 
                "description": "Llama 2 7B Chat - Meta's conversational model",
                "pull_command": "ollama pull llama2:7b-chat"
            },
            "phi": {
                "size": "2.7B",
                "description": "Microsoft Phi-2 - Small but capable",
                "pull_command": "ollama pull phi"
            },
            "neural-chat": {
                "size": "7B",
                "description": "Intel's Neural Chat - Optimized for conversations",
                "pull_command": "ollama pull neural-chat"
            },
            "mixtral:instruct": {
                "size": "47B",
                "description": "Mixtral 8x7B - Powerful MoE model",
                "pull_command": "ollama pull mixtral:instruct"
            },
            "gemma:7b-instruct": {
                "size": "7B",
                "description": "Google's Gemma 7B Instruct",
                "pull_command": "ollama pull gemma:7b-instruct"
            }
        }