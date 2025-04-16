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
        # Create Azure AI Agent instance
        self._agent = await Config.CreateAzureAIAgent(
            kernel=self._kernel,
            agent_name=self._agent_name,
            instructions=self._system_message
        )
        
        # Register tools with the agent
        for tool in self._tools:
            self._agent.add_function(tool)
            
        return self

    def _default_system_message(self) -> str:
        """Return a default system message for this agent type."""
        return f"You are an AI assistant named {self._agent_name}. Help the user by providing accurate and helpful information."

    def _register_functions(self):
        """Register this agent's functions with the kernel."""
        # Register the action handler as a kernel function
        self._kernel.add_function(
            self.handle_action_request, 
            plugin_name=self._agent_name, 
            function_name="handle_action_request"
        )

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
        for tool in config.get("tools", []):
            try:
                function_name = tool["name"]
                description = tool.get("description", "")
                
                # Create a prompt template for the function
                from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
                
                # Use a minimal prompt based on the tool name and description
                default_prompt = f"""You are performing the {function_name} function.
Description: {description}

User input: {{$input}}

Provide a helpful response."""
                
                # Create a prompt template config
                prompt_config = PromptTemplateConfig(
                    template=default_prompt,
                    name=function_name,
                    description=description
                )
                
                # Create the function WITHOUT specifying function_name (it's in the config)
                # This avoids the duplicate parameter error
                plugin_name = f"{agent_type}_plugin"
                function = KernelFunction.from_prompt(
                    prompt_config,
                    plugin_name=plugin_name,
                    description=description
                )
                
                # Add to our list
                kernel_functions.append(function)
                
            except Exception as e:
                logging.warning(f"Failed to create tool '{tool.get('name', 'unknown')}': {e}")
                
        return kernel_functions

    async def handle_action_request(
        self, action_request_json: str
    ) -> str:
        """Handle an action request from another agent or the system."""
        # Ensure the agent is initialized
        if self._agent is None:
            await self.async_init()
            
        try:
            action_request = ActionRequest.parse_raw(action_request_json)
            
            step: Optional[Step] = await self._memory_store.get_step(
                action_request.step_id, action_request.session_id
            )
            
            if not step:
                response = ActionResponse(
                    step_id=action_request.step_id,
                    status=StepStatus.failed,
                    message="Step not found in memory."
                )
                return response.json()
            
            try:
                # Execute using Azure AI Agent
                result_content = await self._execute_with_azure_ai_agent(step, action_request)

                # Store agent message in cosmos
                await self._memory_store.add_item(
                    AgentMessage(
                        session_id=action_request.session_id,
                        user_id=self._user_id,
                        plan_id=action_request.plan_id,
                        content=f"{result_content}",
                        source=self._agent_name,
                        step_id=action_request.step_id,
                    )
                )
                
                track_event_if_configured(
                    "Base agent - Added into the cosmos",
                    {
                        "session_id": action_request.session_id,
                        "user_id": self._user_id,
                        "plan_id": action_request.plan_id,
                        "content": f"{result_content}",
                        "source": self._agent_name,
                        "step_id": action_request.step_id,
                    },
                )
                
                # Update the step
                step.status = StepStatus.completed
                step.agent_reply = result_content
                await self._memory_store.update_step(step)
                
                track_event_if_configured(
                    "Base agent - Updated step and updated into the cosmos",
                    {
                        "status": StepStatus.completed,
                        "session_id": action_request.session_id,
                        "agent_reply": f"{result_content}",
                        "user_id": self._user_id,
                        "plan_id": action_request.plan_id,
                        "content": f"{result_content}",
                        "source": self._agent_name,
                        "step_id": action_request.step_id,
                    },
                )
                
                # Create the response
                action_response = ActionResponse(
                    step_id=step.id,
                    plan_id=step.plan_id,
                    session_id=action_request.session_id,
                    result=result_content,
                    status=StepStatus.completed,
                )
                
                # Publish to group chat manager
                await self._publish_to_group_chat_manager(action_response)
                
                return action_response.json()
                
            except Exception as e:
                logging.exception(f"Error during agent execution: {e}")
                track_event_if_configured(
                    "Base agent - Error during execution, captured into the cosmos",
                    {
                        "session_id": action_request.session_id,
                        "user_id": self._user_id,
                        "plan_id": action_request.plan_id,
                        "content": f"{e}",
                        "source": self._agent_name,
                        "step_id": action_request.step_id,
                    },
                )
                
                error_response = ActionResponse(
                    step_id=action_request.step_id,
                    plan_id=action_request.plan_id,
                    session_id=action_request.session_id,
                    status=StepStatus.failed,
                    message=str(e)
                )
                return error_response.json()
                
        except Exception as e:
            logging.exception(f"Error handling action request: {e}")
            return ActionResponse(
                step_id="unknown",
                status=StepStatus.failed,
                message=f"Error handling action request: {str(e)}"
            ).json()

    async def _execute_with_azure_ai_agent(self, step: Step, action_request: ActionRequest) -> str:
        """Execute the request using Azure AI Agent.
        
        Args:
            step: The step to execute
            action_request: The action request
            
        Returns:
            The result content
        """
        # Ensure the agent is initialized
        if self._agent is None:
            await self.async_init()
            
        # Create chat history for the agent
        messages = []
        
        if step.human_feedback:
            messages.append({
                "role": "user", 
                "content": f"Task: {action_request.action}\n\nHuman feedback: {step.human_feedback}"
            })
        else:
            messages.append({
                "role": "user", 
                "content": f"Task: {action_request.action}\n\nPlease complete this task."
            })
        
        # Pass context to the agent execution
        execution_settings = {
            "step_id": action_request.step_id,
            "session_id": action_request.session_id,
            "plan_id": action_request.plan_id,
            "action": action_request.action,
        }
        
        # Execute the agent with the messages
        result = await self._agent.invoke_async(
            messages=messages,
            kernel_arguments=KernelArguments(**execution_settings)
        )
        
        # Extract the result content
        return result.value

    async def _execute_with_function_calling(self, step: Step, action_request: ActionRequest) -> str:
        """Execute the request using regular function calling.
        
        Args:
            step: The step to execute
            action_request: The action request
            
        Returns:
            The result content
        """
        # Update chat history
        self._chat_history.extend([
            {"role": "assistant", "content": action_request.action, "name": "GroupChatManager"},
            {"role": "user", "content": f"{step.human_feedback or 'Please complete this task'}. Now make the function call", "name": "HumanAgent"},
        ])
        
        # Set up variables
        variables = KernelArguments()
        variables["step_id"] = action_request.step_id
        variables["session_id"] = action_request.session_id
        variables["plan_id"] = action_request.plan_id
        variables["action"] = action_request.action
        variables["chat_history"] = str(self._chat_history)
        
        # Execute with LLM planner
        planner = self._kernel.func("planner", "execute_with_tool")
        
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self._tools])
        variables["tools"] = tool_descriptions
        
        plan_result = await planner.invoke_async(variables=variables)
        tool_name = plan_result.result.strip()
        
        selected_tool = next((t for t in self._tools if t.name == tool_name), None)
        if not selected_tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool_result = await selected_tool.invoke_async(variables=variables)
        return tool_result.result

    async def _publish_to_group_chat_manager(self, response: ActionResponse) -> None:
        """Publish a message to the group chat manager."""
        group_chat_manager_id = f"group_chat_manager_{self._session_id}"
        
        if hasattr(self._kernel, 'get_service'):
            connector = self._kernel.get_service(group_chat_manager_id)
            if connector:
                await connector.invoke_async(response.json())
        else:
            logging.warning(f"No connector service found for {group_chat_manager_id}")

    def save_state(self) -> Mapping[str, Any]:
        """Save agent state for persistence."""
        return {"memory": self._memory_store.save_state()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Load agent state from persistence."""
        self._memory_store.load_state(state["memory"])