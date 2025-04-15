import logging
from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

async def dummy_function() -> str:
    """This is a placeholder function, for a proper Azure AI Search RAG process."""
    return "This is a placeholder function"

# Create the GenericTools function
def get_generic_tools(kernel: sk.Kernel) -> List[KernelFunction]:
    """Get the list of tools available for the Generic Agent."""
    # Convert the function to a kernel function
    dummy_kernel_function = kernel.register_native_function(
        function=dummy_function,
        name="dummy_function",
        description="This is a placeholder"
    )
    
    # Return the list of kernel functions
    return [dummy_kernel_function]

class GenericAgent(BaseAgent):
    """Generic agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        generic_tools: List[KernelFunction],
    ) -> None:
        """Initialize the Generic Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            generic_tools: List of tools available to this agent
        """
        super().__init__(
            agent_name="GenericAgent",
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=generic_tools,
            system_message="You are a generic agent. You are used to handle generic tasks that a general Large Language Model can assist with. You are being called as a fallback, when no other agents are able to use their specialised functions in order to solve the user's task. Summarize back the user what was done. Do not use any function calling- just use your native LLM response."
        )