from typing import List, Optional

import semantic_kernel as sk
from context.cosmos_memory_kernel import CosmosMemoryContext
from kernel_agents.agent_base import BaseAgent
from kernel_tools.marketing_tools import MarketingTools
from models.messages_kernel import AgentType
from semantic_kernel.functions import KernelFunction


class MarketingAgent(BaseAgent):
    """Marketing agent implementation using Semantic Kernel.

    This agent specializes in marketing, campaign management, and analyzing market data.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.MARKETING.value,
        config_path: Optional[str] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Marketing Agent.

        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "MarketingAgent")
            config_path: Optional path to the Marketing tools configuration file
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if not tools:
            # Get tools directly from MarketingTools class
            tools_dict = MarketingTools.get_all_kernel_functions()
            tools = [KernelFunction.from_method(func) for func in tools_dict.values()]

            # Load the marketing tools configuration for system message
            config = self.load_tools_config("marketing", config_path)

            # Use system message from config if not explicitly provided
            if not system_message:
                system_message = config.get(
                    "system_message",
                    "You are an AI Agent. You have knowledge about marketing, including campaigns, market research, and promotional activities.",
                )

            # Use agent name from config if available
            agent_name = AgentType.MARKETING.value

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
        """Get the plugins for the marketing agent."""
        return MarketingTools.get_all_kernel_functions()
