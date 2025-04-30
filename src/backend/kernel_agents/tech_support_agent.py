from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import AgentType
from src.backend.kernel_tools.tech_support_tools import TechSupportTools


class TechSupportAgent(BaseAgent):
    """Tech Support agent implementation using Semantic Kernel.

    This agent specializes in technical support, IT administration, and equipment setup.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.TECH_SUPPORT.value,
        config_path: Optional[str] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Tech Support Agent.

        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "TechSupportAgent")
            config_path: Optional path to the Tech Support tools configuration file
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if tools is None:
            # Get tools directly from TechSupportTools class
            tools_dict = TechSupportTools.get_all_kernel_functions()
            tools = [KernelFunction.from_method(func) for func in tools_dict.values()]

            # Load the tech support tools configuration for system message
            config = self.load_tools_config("tech_support", config_path)

            # Use system message from config if not explicitly provided
            if not system_message:
                system_message = config.get(
                    "system_message",
                    "You are a Tech Support agent. You can assist with technical issues, IT administration, equipment setup, and software/hardware troubleshooting. When asked to call a function, you should summarize back what was done.",
                )

            # Use agent name from config if available
            agent_name = AgentType.TECH_SUPPORT.value

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
        """Get the plugins for the tech support agent."""
        return TechSupportTools.get_all_kernel_functions()
