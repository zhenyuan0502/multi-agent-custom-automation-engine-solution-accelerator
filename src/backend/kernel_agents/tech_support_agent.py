from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from multi_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

class TechSupportAgent(BaseAgent):
    """Tech Support agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tech_support_tools: List[KernelFunction],
        config_path: Optional[str] = None
    ) -> None:
        """Initialize the Tech Support Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tech_support_tools: List of tools available to this agent
            config_path: Optional path to the tech support tools configuration file
        """
        # Load configuration
        config = self.load_tools_config("tech_support", config_path)
        
        super().__init__(
            agent_name=config.get("agent_name", "TechSupportAgent"),
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tech_support_tools,
            system_message=config.get("system_message", "You are a Tech Support agent. You help users resolve technology-related problems.")
        )