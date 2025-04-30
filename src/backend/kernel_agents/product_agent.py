from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import AgentType
from kernel_tools.product_tools import ProductTools


class ProductAgent(BaseAgent):
    """Product agent implementation using Semantic Kernel.

    This agent specializes in product management, development, and related tasks.
    It can provide information about products, manage inventory, handle product
    launches, analyze sales data, and coordinate with other teams like marketing
    and tech support.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.PRODUCT.value,
        config_path: Optional[str] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Product Agent.

        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "ProductAgent")
            config_path: Optional path to the Product tools configuration file
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if not tools:
            # Get tools directly from ProductTools class
            tools_dict = ProductTools.get_all_kernel_functions()
            tools = [KernelFunction.from_method(func) for func in tools_dict.values()]

            # Load the product tools configuration for system message
            config = self.load_tools_config("product", config_path)

            # Use system message from config if not explicitly provided
            if not system_message:
                system_message = config.get(
                    "system_message",
                    "You are a Product agent. You have knowledge about product management, development, and compliance guidelines. When asked to call a function, you should summarize back what was done.",
                )

            # Use agent name from config if available
            agent_name = AgentType.PRODUCT.value

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
        """Get the plugins for the product agent."""
        return ProductTools.get_all_kernel_functions()

    # Explicitly inherit handle_action_request from the parent class
    # This is not technically necessary but makes the inheritance explicit
    async def handle_action_request(self, action_request_json: str) -> str:
        """Handle an action request from another agent or the system.

        This method is inherited from BaseAgent but explicitly included here for clarity.

        Args:
            action_request_json: The action request as a JSON string

        Returns:
            A JSON string containing the action response
        """
        return await super().handle_action_request(action_request_json)
