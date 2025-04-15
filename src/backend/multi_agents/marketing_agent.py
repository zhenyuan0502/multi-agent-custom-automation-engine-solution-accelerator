from typing import List, Dict, Any, Optional
import json
import os

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.kernel_arguments import KernelArguments

from multi_agents.semantic_kernel_agent import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown. Then write AGENT SUMMARY: and then include a summary of what you did."

# Define a dynamic function creator
def create_dynamic_function(name: str, response_template: str, formatting_instr: str = formatting_instructions):
    """Create a dynamic function for marketing tools based on the name and template."""
    async def dynamic_function(*args, **kwargs) -> str:
        try:
            # Format the template with the provided kwargs
            return response_template.format(**kwargs) + f"\n{formatting_instr}"
        except KeyError as e:
            return f"Error: Missing parameter {e} for {name}"
        except Exception as e:
            return f"Error processing {name}: {str(e)}"
    
    # Set the function name
    dynamic_function.__name__ = name
    return dynamic_function

# Function to load tools from JSON configuration
def load_marketing_tools_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load marketing tools configuration from a JSON file."""
    if config_path is None:
        # Default path relative to the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(current_dir))
        config_path = os.path.join(backend_dir, "tools", "marketing_tools.json")
    
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading marketing tools configuration: {e}")
        # Return empty default configuration
        return {"agent_name": "MarketingAgent", "system_message": "", "tools": []}

# Create the marketing tools function that loads from JSON
def get_marketing_tools(kernel: sk.Kernel, config_path: Optional[str] = None) -> List[KernelFunction]:
    """Get the list of marketing tools for the Marketing Agent from configuration."""
    # Load configuration
    config = load_marketing_tools_config(config_path)
    
    # Convert the configured tools to kernel functions
    kernel_functions = []
    for tool in config.get("tools", []):
        # Create the dynamic function
        func = create_dynamic_function(
            name=tool["name"],
            response_template=tool.get("response_template", "")
        )
        
        # Register with the kernel
        kernel_function = kernel.register_native_function(
            function=func,
            name=tool["name"],
            description=tool.get("description", "")
        )
        kernel_functions.append(kernel_function)
    
    return kernel_functions

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