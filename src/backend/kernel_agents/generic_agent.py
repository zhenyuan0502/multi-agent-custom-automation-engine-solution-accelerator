import logging
from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

# Define Generic tools (functions)
@kernel_function(
    description="Get current date and time",
    name="get_current_datetime"
)
async def get_current_datetime() -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@kernel_function(
    description="Perform simple calculations",
    name="calculate"
)
async def calculate(expression: str) -> str:
    """Perform simple calculations."""
    import re
    # Validate the expression to ensure it contains only allowed characters
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
        return "Error: Expression contains invalid characters. Only digits and basic operators (+, -, *, /) are allowed."
    
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

# Create the GenericTools function
def get_generic_tools(kernel: sk.Kernel) -> List[KernelFunction]:
    """Get the list of generic tools for the Generic Agent."""
    # Define all generic functions
    generic_functions = [
        get_current_datetime,
        calculate
    ]
    
    # Register each function with the kernel and collect KernelFunction objects
    kernel_functions = []
    for func in generic_functions:
        kernel.add_function(func, plugin_name="generic")
        kernel_functions.append(kernel.get_function(plugin_name="generic", function_name=func.__name__))
    
    return kernel_functions

class GenericAgent(BaseAgent):
    """Generic agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: List[KernelFunction] = None,
        system_message: Optional[str] = None,
        agent_name: str = "GenericAgent",
        config_path: Optional[str] = None
    ) -> None:
        """Initialize the Generic Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "GenericAgent")
            config_path: Optional path to the Generic tools configuration file
        """
        # Load configuration if tools not provided
        if tools is None:
            # For generic agent, we prefer using the hardcoded tools
            tools = get_generic_tools(kernel)
            # But also load configuration for system message and name
            config = self.load_tools_config("generic", config_path)
            if not system_message:
                system_message = config.get("system_message", "You are a helpful assistant capable of performing general tasks.")
            agent_name = config.get("agent_name", agent_name)
        
        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message
        )