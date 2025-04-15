from typing import List, Dict, Any, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.kernel_arguments import KernelArguments

from multi_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

def load_marketing_tools_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load marketing tools configuration from a JSON file."""
    return BaseAgent.load_tools_config("marketing", config_path)

def get_marketing_tools(kernel: sk.Kernel, config_path: Optional[str] = None) -> List[KernelFunction]:
    """Get the list of marketing tools for the Marketing Agent from configuration."""
    return BaseAgent.get_tools_from_config(kernel, "marketing", config_path)

class MarketingAgent(BaseAgent):
    """Marketing agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        marketing_tools: List[KernelFunction],
        config_path: Optional[str] = None
    ) -> None:
        """Initialize the Marketing Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            marketing_tools: List of tools available to this agent
            config_path: Optional path to the marketing tools configuration file
        """
        # Load configuration
        config = load_marketing_tools_config(config_path)
        
        super().__init__(
            agent_name=config.get("agent_name", "MarketingAgent"),
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=marketing_tools,
            system_message=config.get("system_message", "You are a Marketing agent. You specialize in marketing strategy, campaign development, content creation, and market analysis.")
        )