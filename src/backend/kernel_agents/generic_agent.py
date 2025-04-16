import logging
from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

# Use the kernel_function decorator for proper registration
@kernel_function(
    description="This is a placeholder function, for a proper Azure AI Search RAG process.",
    name="dummy_function"
)
async def dummy_function() -> str:
    """This is a placeholder function, for a proper Azure AI Search RAG process."""
    return "This is a placeholder function"

# Create the GenericTools function
def get_generic_tools(kernel: sk.Kernel) -> List[KernelFunction]:
    """Get the list of tools available for the Generic Agent."""
    # Register the function with the kernel and get it back as a kernel function
    kernel.add_function(dummy_function)
    # Return the list of registered functions
    return [kernel.get_function("dummy_function")]

class GenericAgent(BaseAgent):
    """Generic agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: List[KernelFunction] = None,
        agent_name: str = "GenericAgent",
        system_message: str = None,
        **kwargs
    ) -> None:
        """Initialize the Generic Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent
            agent_name: The name of the agent
            system_message: The system message for this agent
            **kwargs: Additional arguments
        """
        default_system_message = "You are a generic agent. You are used to handle generic tasks that a general Large Language Model can assist with. You are being called as a fallback, when no other agents are able to use their specialised functions in order to solve the user's task. Summarize back the user what was done. Do not use any function calling- just use your native LLM response."
        
        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools or [],
            system_message=system_message or default_system_message
        )