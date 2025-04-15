from typing import List, Dict, Any, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.kernel_arguments import KernelArguments
from typing_extensions import Annotated

from multi_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

def load_hr_tools_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load HR tools configuration from a JSON file."""
    return BaseAgent.load_tools_config("hr", config_path)

def get_hr_tools(kernel: sk.Kernel, config_path: Optional[str] = None) -> List[KernelFunction]:
    """Get the list of HR tools for the HR Agent from configuration."""
    return BaseAgent.get_tools_from_config(kernel, "hr", config_path)

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
        config = load_hr_tools_config(config_path)
        
        super().__init__(
            agent_name=config.get("agent_name", "HrAgent"),
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=hr_tools,
            system_message=config.get("system_message", "You are an AI Agent. You have knowledge about HR policies, procedures, and onboarding guidelines.")
        )