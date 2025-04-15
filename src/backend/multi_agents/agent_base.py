import logging
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Callable, Awaitable

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.kernel_arguments import KernelArguments
# Import core components needed for Semantic Kernel plugins
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter


# Import Pydantic model base
from semantic_kernel.kernel_pydantic import KernelBaseModel

from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    ActionRequest,
    ActionResponse,
    AgentMessage,
    Step,
    StepStatus,
)
from event_utils import track_event_if_configured

# Default formatting instructions used across agents
DEFAULT_FORMATTING_INSTRUCTIONS = "Instructions: returning the output of this function call verbatim to the user in markdown. Then write AGENT SUMMARY: and then include a summary of what you did."

class BaseAgent(KernelBaseModel):
    """BaseAgent implemented using Semantic Kernel instead of AutoGen."""

    def __init__(
        self,
        agent_name: str,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: List[KernelFunction],
        system_message: str,
    ):
        super().__init__()
        self._agent_name = agent_name
        self._kernel = kernel
        self._session_id = session_id
        self._user_id = user_id
        self._memory_store = memory_store
        self._tools = tools
        self._system_message = system_message
        self._chat_history = [{"role": "system", "content": system_message}]
        
        self._register_functions()

    def _register_functions(self):
        """Register this agent's functions with the kernel."""
        # Register the action handler as a native function
        self._kernel.import_skill(self, skill_name=self._agent_name)

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
            # Default path relative to the caller's file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(backend_dir, "tools", f"{agent_type}_tools.json")
        
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {agent_type} tools configuration: {e}")
            # Return empty default configuration
            return {
                "agent_name": f"{agent_type.capitalize()}Agent",
                "system_message": "",
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
            # Create the dynamic function
            func = cls.create_dynamic_function(
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

    @kernel_function(
        description="Handle an action request from another agent",
        name="handle_action_request",
    )
    @kernel_function_context_parameter(
        name="action_request_json",
        description="JSON string of the action request",
    )
    async def handle_action_request(
        self, context: KernelArguments
    ) -> str:
        """Handle an action request from another agent or the system."""
        try:
            action_request_json = context["action_request_json"]
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
            
            self._chat_history.extend([
                {"role": "assistant", "content": action_request.action, "name": "GroupChatManager"},
                {"role": "user", "content": f"{step.human_feedback}. Now make the function call", "name": "HumanAgent"},
            ])
            
            try:
                variables = KernelArguments()
                variables["step_id"] = action_request.step_id
                variables["session_id"] = action_request.session_id
                variables["plan_id"] = action_request.plan_id
                variables["action"] = action_request.action
                variables["chat_history"] = str(self._chat_history)
                
                result = await self._execute_tool_with_llm(variables)
                
                await self._memory_store.add_item(
                    AgentMessage(
                        session_id=action_request.session_id,
                        user_id=self._user_id,
                        plan_id=action_request.plan_id,
                        content=f"{result}",
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
                        "content": f"{result}",
                        "source": self._agent_name,
                        "step_id": action_request.step_id,
                    },
                )
                
                step.status = StepStatus.completed
                step.agent_reply = result
                await self._memory_store.update_step(step)
                
                track_event_if_configured(
                    "Base agent - Updated step and updated into the cosmos",
                    {
                        "status": StepStatus.completed,
                        "session_id": action_request.session_id,
                        "agent_reply": f"{result}",
                        "user_id": self._user_id,
                        "plan_id": action_request.plan_id,
                        "content": f"{result}",
                        "source": self._agent_name,
                        "step_id": action_request.step_id,
                    },
                )
                
                action_response = ActionResponse(
                    step_id=step.id,
                    plan_id=step.plan_id,
                    session_id=action_request.session_id,
                    result=result,
                    status=StepStatus.completed,
                )
                
                await self._publish_to_group_chat_manager(action_response)
                
                return action_response.json()
                
            except Exception as e:
                logging.exception(f"Error during tool execution: {e}")
                track_event_if_configured(
                    "Base agent - Error during tool execution, captured into the cosmos",
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

    async def _execute_tool_with_llm(self, variables: KernelArguments) -> str:
        """Execute the appropriate tool based on LLM reasoning."""
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
        return {"memory": self._memory_store.save_state()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self._memory_store.load_state(state["memory"])