from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

class HrAgent(BaseAgent):
    """HR agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        hr_tools: List[KernelFunction],
        config_path: Optional[str] = None
    ) -> None:
        """Initialize the HR Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            hr_tools: List of tools available to this agent
            config_path: Optional path to the HR tools configuration file
        """
        # Load configuration
        config = self.load_tools_config("hr", config_path)
        
        super().__init__(
            agent_name=config.get("agent_name", "HrAgent"),
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=hr_tools,
            system_message=config.get("system_message", "You are an AI Agent. You have knowledge about HR policies, procedures, and onboarding guidelines.")
        )