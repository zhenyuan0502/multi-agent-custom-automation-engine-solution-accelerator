from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

class ProductAgent(BaseAgent):
    """Product agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        product_tools: List[KernelFunction],
        config_path: Optional[str] = None
    ) -> None:
        """Initialize the Product Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            product_tools: List of tools available to this agent
            config_path: Optional path to the Product tools configuration file
        """
        # Load configuration
        config = self.load_tools_config("product", config_path)
        
        super().__init__(
            agent_name="ProductAgent",
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=product_tools,
            system_message=config.get("system_message", "You are a Product agent. You have knowledge about products, their specifications, pricing, availability, and features. You can provide detailed information about products, compare them, and manage product data.")
        )