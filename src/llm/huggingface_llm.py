"""
HuggingFace LLM provider implementation using open-source models.
"""
import os
import torch
import asyncio
from typing import List, Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from huggingface_hub import InferenceClient
from .base import BaseLLMProvider


class HuggingFaceLLMProvider(BaseLLMProvider):
    """HuggingFace LLM provider for open-source language models."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_api: bool = False,
        api_token: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_tokens: int = 4096
    ):
        """
        Initialize HuggingFace LLM provider.
        
        Args:
            model_name: HuggingFace model name
            use_api: Use HuggingFace Inference API instead of local
            api_token: HuggingFace API token (required for API)
            device: Device to run on (cuda, cpu, mps)
            load_in_4bit: Load model in 4-bit quantization
            load_in_8bit: Load model in 8-bit quantization
            max_tokens: Maximum tokens for generation
        """
        super().__init__(model_name, max_tokens)
        
        self.use_api = use_api
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_api:
            # Initialize API client
            self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
            if not self.api_token:
                raise ValueError("HuggingFace API token required for API usage")
            
            self.client = InferenceClient(token=self.api_token)
        else:
            # Load model locally
            print(f"Loading model {model_name} on {self.device}...")
            
            # Configure quantization if requested
            quantization_config = None
            if load_in_4bit or load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if self.device != "cuda" and not (load_in_4bit or load_in_8bit):
                self.model = self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device != "cuda" else None
            )
            
            print(f"Model loaded successfully on {self.device}")
        
        # Set metadata
        self.metadata = {
            "use_api": use_api,
            "device": self.device,
            "quantization": "4bit" if load_in_4bit else "8bit" if load_in_8bit else "none"
        }
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using HuggingFace model."""
        max_tokens = max_tokens or self.max_tokens
        
        # Format prompt with system prompt if provided
        if system_prompt:
            full_prompt = self._format_prompt_with_system(prompt, system_prompt)
        else:
            full_prompt = prompt
        
        if self.use_api:
            # Use API
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.text_generation(
                        full_prompt,
                        model=self.model_name,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        return_full_text=False
                    )
                )
                return response
            except Exception as e:
                print(f"API generation error: {e}")
                return ""
        else:
            # Use local model
            try:
                # Run generation in executor to avoid blocking
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.pipeline(
                        full_prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        return_full_text=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                )
                return response[0]["generated_text"]
            except Exception as e:
                print(f"Local generation error: {e}")
                return ""
    
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """Generate text for multiple prompts."""
        # For now, process sequentially
        # TODO: Implement true batch processing for local models
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, system_prompt, temperature, max_tokens)
            results.append(result)
        return results
    
    def get_cost_per_token(self) -> Dict[str, float]:
        """Get cost per token (free for local, minimal for API)."""
        if self.use_api:
            # HuggingFace Inference API pricing (approximate)
            return {
                "input": 0.00001 / 1000,
                "output": 0.00001 / 1000
            }
        else:
            # Free for local inference
            return {
                "input": 0.0,
                "output": 0.0
            }
    
    def _format_prompt_with_system(self, prompt: str, system_prompt: str) -> str:
        """Format prompt with system instructions based on model type."""
        # Different models have different prompt formats
        if "mistral" in self.model_name.lower():
            return f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        elif "llama" in self.model_name.lower():
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        elif "phi" in self.model_name.lower():
            return f"Instruct: {system_prompt}\n\n{prompt}\nOutput:"
        elif "gemma" in self.model_name.lower():
            return f"<start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Generic format
            return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
    
    @staticmethod
    def get_recommended_models() -> Dict[str, Dict[str, Any]]:
        """Get list of recommended open-source models."""
        return {
            "mistralai/Mistral-7B-Instruct-v0.2": {
                "size": "7B",
                "description": "Fast, efficient instruction-following model",
                "min_gpu_memory": "16GB",
                "quantized_memory": "6GB"
            },
            "microsoft/Phi-3-mini-4k-instruct": {
                "size": "3.8B",
                "description": "Small but capable model, good for limited resources",
                "min_gpu_memory": "8GB",
                "quantized_memory": "4GB"
            },
            "meta-llama/Llama-2-7b-chat-hf": {
                "size": "7B",
                "description": "Meta's Llama 2 chat model",
                "min_gpu_memory": "16GB",
                "quantized_memory": "6GB"
            },
            "google/gemma-7b-it": {
                "size": "7B",
                "description": "Google's instruction-tuned Gemma model",
                "min_gpu_memory": "16GB",
                "quantized_memory": "6GB"
            },
            "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ": {
                "size": "46.7B",
                "description": "Powerful MoE model (quantized)",
                "min_gpu_memory": "24GB",
                "quantized_memory": "24GB"
            }
        }