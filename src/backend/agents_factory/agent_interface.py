"""Base class for all agents in the Multi-Agent Custom Automation Engine."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction
from semantic_kernel.memory import MemoryStore

from agents_factory.agent_config import AgentBaseConfig

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the Multi-Agent Custom Automation Engine."""

    def __init__(
        self,
        config: AgentBaseConfig,
        tools: Optional[List[KernelFunction]] = None,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        **kwargs
    ):
        """Initialize the base agent.

        Args:
            config: The configuration for the agent
            tools: Optional list of tools (kernel functions) to add to the agent
            temperature: The temperature parameter for the model
            system_message: Optional system message for the agent
            **kwargs: Additional parameters for specific agent implementations
        """
        self.config = config
        self.tools = tools or []
        self.temperature = temperature
        self.system_message = system_message
        self.kernel = config.kernel
        self.memory_store = config.memory_store
        self.session_id = config.session_id
        self.user_id = config.user_id
        
        # Additional properties can be set from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize the agent (register tools, etc.)
        self._initialize()
        
    def _initialize(self) -> None:
        """Initialize the agent by registering tools and other setup tasks."""
        # Register all tools with the kernel
        for tool in self.tools:
            if tool and not self.kernel.has_function(tool.name):
                self.kernel.add_function(tool)
                logger.debug(f"Registered tool {tool.name} for agent")
        
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message and generate a response.
        
        Args:
            message: The input message containing user input and context
            
        Returns:
            A response message
        """
        pass
        
    async def remember(self, key: str, value: Any, description: Optional[str] = None) -> None:
        """Save information to the agent's memory.
        
        Args:
            key: The key to store the information under
            value: The value to store
            description: Optional description of the memory
        """
        if self.memory_store:
            # Format a unique ID for this memory based on session and key
            memory_id = f"{self.session_id}:{key}"
            await self.memory_store.save_information(memory_id, value, description or key)
            logger.debug(f"Saved memory with key {key} for session {self.session_id}")
            
    async def recall(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Recall information from the agent's memory based on a query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            A list of memory items matching the query
        """
        if self.memory_store:
            # Include session ID in the search to scope to this session
            search_query = f"{self.session_id} {query}"
            results = await self.memory_store.search(search_query, limit=limit)
            return [
                {
                    "text": result.text,
                    "description": result.description,
                    "relevance": result.relevance
                }
                for result in results
            ]
        return []
        
    async def clear_memory(self) -> None:
        """Clear the agent's memory for the current session."""
        if self.memory_store:
            # Get all memories for this session
            session_memories = await self.recall(self.session_id, limit=100)
            for memory in session_memories:
                # Delete each memory
                await self.memory_store.remove(memory["text"])
            logger.info(f"Cleared all memories for session {self.session_id}")
            
    def get_system_message(self) -> str:
        """Get the system message for this agent, including role-specific instructions.
        
        Returns:
            The complete system message for the agent
        """
        # Start with the base system message if provided
        base_message = self.system_message or "You are an AI assistant helping with a task."
        
        # Add agent-specific instructions (should be implemented by subclasses)
        role_instructions = self._get_role_instructions()
        
        # Combine them
        return f"{base_message}\n\n{role_instructions}"
        
    def _get_role_instructions(self) -> str:
        """Get role-specific instructions for this agent type.
        
        This should be overridden by subclasses to provide specific guidance for different agent types.
        
        Returns:
            Role-specific instructions as a string
        """
        return "As an AI assistant, provide helpful, accurate, and relevant information to the user's request."