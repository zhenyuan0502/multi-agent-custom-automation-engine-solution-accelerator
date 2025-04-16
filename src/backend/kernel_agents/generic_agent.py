import logging
from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

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
            config = self.load_tools_config("generic", config_path)
            tools = self.get_tools_from_config(kernel, "generic", config_path)
            if not system_message:
                system_message = config.get("system_message", 
                    "You are a generic agent. You are used to handle generic tasks that a general Large Language Model can assist with. "
                    "You are being called as a fallback, when no other agents are able to use their specialised functions in order to solve "
                    "the user's task. Summarize back to the user what was done.")
            agent_name = config.get("agent_name", agent_name)
        
        # Call the parent initializer with the agent_type parameter to ensure proper tool loading
        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            agent_type="generic"  # Explicitly provide the agent_type for proper initialization
        )
        
        # NOTE: We're removing the duplicate registration here because BaseAgent._register_functions()
        # already registers the handle_action_request function with the kernel
    
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