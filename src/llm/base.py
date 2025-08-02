"""
Base class for LLM providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import asyncio


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model_name: str, max_tokens: int = 4096):
        """
        Initialize the LLM provider.
        
        Args:
            model_name: Name of the model
            max_tokens: Maximum tokens for generation
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.metadata = {}
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def generate_batch(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated texts
        """
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> Dict[str, float]:
        """
        Get the cost per token for input and output.
        
        Returns:
            Dictionary with 'input' and 'output' costs per token
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the LLM provider.
        
        Returns:
            Dictionary containing provider metadata
        """
        return {
            "provider": self.__class__.__name__,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "cost_per_token": self.get_cost_per_token(),
            **self.metadata
        }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a chat completion from messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        system_prompt = None
        
        # Extract system prompt if present
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
            prompt = self._messages_to_prompt(messages)
        
        return await self.generate(prompt, system_prompt, temperature, max_tokens)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)