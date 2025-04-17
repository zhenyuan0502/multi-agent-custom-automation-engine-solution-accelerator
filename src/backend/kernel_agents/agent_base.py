import logging
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Callable, Awaitable

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent

from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    ActionRequest,
    ActionResponse,
    AgentMessage,
    Step,
    StepStatus,
)
from config_kernel import Config
from event_utils import track_event_if_configured

# Default formatting instructions used across agents
DEFAULT_FORMATTING_INSTRUCTIONS = "Instructions: returning the output of this function call verbatim to the user in markdown. Then write AGENT SUMMARY: and then include a summary of what you did."

class BaseAgent:
    """BaseAgent implemented using Semantic Kernel with Azure AI Agent support."""

    def __init__(
        self,
        agent_name: str,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_type: Optional[str] = None,
    ):
        """Initialize the base agent.
        
        Args:
            agent_name: The name of the agent
            kernel: The semantic kernel instance
            session_id: The session ID
            user_id: The user ID
            memory_store: The memory context for storing agent state
            tools: Optional list of tools for the agent
            system_message: Optional system message for the agent
            agent_type: Optional agent type string for automatic tool loading
        """
        self._agent_name = agent_name
        self._kernel = kernel
        self._session_id = session_id
        self._user_id = user_id
        self._memory_store = memory_store
        
        # If agent_type is provided, load tools from config automatically
        if agent_type and not tools:
            self._tools = self.get_tools_from_config(kernel, agent_type)
            
            # If system_message isn't provided, try to get it from config
            if not system_message:
                config = self.load_tools_config(agent_type)
                system_message = config.get("system_message", self._default_system_message())
        else:
            self._tools = tools or []
        
        self._system_message = system_message or self._default_system_message()
        self._chat_history = [{"role": "system", "content": self._system_message}]
        
        # The agent will be created asynchronously in the async_init method
        self._agent = None
        
        # Log initialization
        logging.info(f"Initialized {agent_name} with {len(self._tools)} tools")
        
        # Register the handler functions
        self._register_functions()

    async def async_init(self):
        """Asynchronously initialize the agent after construction.
        
        This method must be called after creating the agent to complete initialization.
        """
        # Create Azure AI Agent or fallback
        self._agent = await Config.CreateAzureAIAgent(
            kernel=self._kernel,
            agent_name=self._agent_name,
            instructions=self._system_message
        )
        # Tools are registered with the kernel via get_tools_from_config
        return self

    def _default_system_message(self) -> str:
        """Return a default system message for this agent type."""
        return f"You are an AI assistant named {self._agent_name}. Help the user by providing accurate and helpful information."

    def _register_functions(self):
        """Register this agent's functions with the kernel."""
        # Use the kernel function decorator approach instead of from_native_method
        # which isn't available in SK 1.28.0
        function_name = "handle_action_request"
        
        # Define the function using the kernel function decorator
        @kernel_function(
            description="Handle an action request from another agent or the system",
            name=function_name
        )
        async def handle_action_request_wrapper(*args, **kwargs):
            # Forward to the instance method
            return await self.handle_action_request(*args, **kwargs)
            
        # Wrap the decorated function into a KernelFunction and register under this agent's plugin
        kernel_func = KernelFunction.from_method(handle_action_request_wrapper)
        # Use agent name as plugin for handler
        self._kernel.add_function(self._agent_name, kernel_func)

    @staticmethod
    def create_dynamic_function(name: str, response_template: str, formatting_instr: str = DEFAULT_FORMATTING_INSTRUCTIONS) -> Callable[..., Awaitable[str]]:
        """Create a dynamic function for agent tools based on the name and template.
        
        Args:
            name: The name of the function to create
            response_template: The template string to use for the response
            formatting_instr: Optional formatting instructions to append to the response
            
        Returns:
            A dynamic async function that can be registered with the semantic kernel
        """
        # Create a dynamic function decorated with @kernel_function
        @kernel_function(
            description=f"Dynamic function: {name}",
            name=name
        )
        async def dynamic_function(*args, **kwargs) -> str:
            try:
                # Format the template with the provided kwargs
                return response_template.format(**kwargs) + f"\n{formatting_instr}"
            except KeyError as e:
                return f"Error: Missing parameter {e} for {name}"
            except Exception as e:
                return f"Error processing {name}: {str(e)}"
        
        return dynamic_function

    @staticmethod
    def load_tools_config(agent_type: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load tools configuration from a JSON file.
        
        Args:
            agent_type: The type of agent (e.g., "marketing", "hr")
            config_path: Optional explicit path to the configuration file
            
        Returns:
            A dictionary containing the configuration
        """
        if config_path is None:
            # Default path relative to the tools directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(current_dir)  # Just one level up to get to backend dir
            config_path = os.path.join(backend_dir, "tools", f"{agent_type}_tools.json")
        
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading {agent_type} tools configuration: {e}")
            # Return empty default configuration
            return {
                "agent_name": f"{agent_type.capitalize()}Agent",
                "system_message": "You are an AI assistant",
                "tools": []
            }

    @classmethod
    def get_tools_from_config(cls, kernel: sk.Kernel, agent_type: str, config_path: Optional[str] = None) -> List[KernelFunction]:
        """Get the list of tools for an agent from configuration.
        
        Args:
            kernel: The semantic kernel instance
            agent_type: The type of agent (e.g., "marketing", "hr")
            config_path: Optional explicit path to the configuration file
            
        Returns:
            A list of KernelFunction objects representing the tools
        """
        # Load configuration
        config = cls.load_tools_config(agent_type, config_path)
        
        # Convert the configured tools to kernel functions
        kernel_functions = []
        plugin_name = f"{agent_type}_plugin"
        
        for tool in config.get("tools", []):
            try:
                function_name = tool["name"]
                description = tool.get("description", "")
                # Create a dynamic function using the JSON response_template
                response_template = tool.get("response_template") or tool.get("prompt_template") or ""
                # Generate a dynamic function matching original agent implementation
                dynamic_fn = cls.create_dynamic_function(function_name, response_template)
                # Wrap and register the dynamic function
                kernel_func = KernelFunction.from_method(dynamic_fn)
                kernel.add_function(plugin_name, kernel_func)
                kernel_functions.append(kernel_func)
                
                logging.info(f"Successfully created dynamic tool '{function_name}' for {agent_type}")
                
            except Exception as e:
                logging.warning(f"Failed to create tool '{tool.get('name', 'unknown')}': {e}")
                
        return kernel_functions