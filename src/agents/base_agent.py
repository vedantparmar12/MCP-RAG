"""
Base agent classes for the self-improvement RAG system.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
from mem0 import Memory

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all self-improvement agents"""
    
    def __init__(self, name: str, memory: Memory):
        self.name = name
        self.memory = memory
        self.state = {"status": "idle", "last_run": None}
        self.logger = logging.getLogger(f"agent.{name}")
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent logic"""
        pass
    
    async def validate_preconditions(self, context: Dict[str, Any]) -> bool:
        """Validate that agent can execute safely"""
        return True
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent with validation and error handling"""
        try:
            # Update state
            self.state["status"] = "running"
            self.state["last_run"] = datetime.now()
            
            # Validate preconditions
            if not await self.validate_preconditions(context):
                return {
                    "agent": self.name,
                    "status": "failed",
                    "error": "Precondition validation failed"
                }
            
            # Execute agent logic
            result = await self.execute(context)
            
            # Store result in memory
            self.memory.add(
                messages=[{
                    "role": "assistant", 
                    "content": f"Agent {self.name} execution result: {result}"
                }],
                metadata={
                    "agent": self.name,
                    "timestamp": str(datetime.now()),
                    "context": context
                }
            )
            
            # Update state
            self.state["status"] = "completed"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent {self.name} failed with error: {str(e)}")
            self.state["status"] = "failed"
            
            return {
                "agent": self.name,
                "status": "failed",
                "error": str(e)
            }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return self.state.copy()