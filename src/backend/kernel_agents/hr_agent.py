from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import AgentType
from kernel_tools.hr_tools import HrTools


class HrAgent(BaseAgent):
    """HR agent implementation using Semantic Kernel.

    This agent provides HR-related functions such as onboarding, benefits management,
    and employee administration.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.HR.value,
        config_path: Optional[str] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the HR Agent.

        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "HrAgent")
            config_path: Optional path to the HR tools configuration file
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if not tools:
            # Get tools directly from HrTools class
            tools_dict = HrTools.get_all_kernel_functions()
            tools = [KernelFunction.from_method(func) for func in tools_dict.values()]

            # Load the HR tools configuration for system message
            config = self.load_tools_config("hr", config_path)

            # Use system message from config if not explicitly provided
            if not system_message:
                system_message = config.get(
                    "system_message",
                    "You are an AI Agent. You have knowledge about HR (e.g., human resources), policies, procedures, and onboarding guidelines.",
                )

            # Use agent name from config if available
            agent_name = AgentType.HR.value

        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            client=client,
            definition=definition,
        )

    @property
    def plugins(self):
        """Get the plugins for the HR agent."""
        return HrTools.get_all_kernel_functions()
