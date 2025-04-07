import logging
from typing import Any, Dict, List, Mapping, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.memory import MemoryRecord
from semantic_kernel.orchestration import SKContext
from semantic_kernel.skill_definition import sk_function

from context.cosmos_memory_kernel import CosmosBufferedMemoryStore
from models.messages_kernel import (
    ActionRequest,
    ActionResponse,
    AgentMessage,
    Step,
    StepStatus,
)
from event_utils import track_event_if_configured


class BaseAgent:
    """BaseAgent implemented using Semantic Kernel instead of AutoGen."""

    def __init__(
        self,
        agent_name: str,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosBufferedMemoryStore,
        tools: List[KernelFunction],
        system_message: str,
    ):
        self._agent_name = agent_name
        self._kernel = kernel
        self._session_id = session_id
        self._user_id = user_id
        self._memory_store = memory_store
        self._tools = tools
        self._system_message = system_message
        self._chat_history = [{"role": "system", "content": system_message}]
        
        # Register the action handler as a semantic function
        self._register_functions()

    def _register_functions(self):
        """Register this agent's functions with the kernel."""
        # Import locally to avoid circular imports
        from semantic_kernel.orchestration.sk_function_decorator import sk_function
        
        # Register the action handler
        self.kernel.register_semantic_function(
            self._agent_name,
            "handle_action_request",
            self.handle_action_request
        )

    @sk_function(
        description="Handle an action request from another agent",
        name="handle_action_request",
    )
    async def handle_action_request(
        self, action_request_json: str, context: SKContext = None
    ) -> str:
        """Handle an action request from another agent or the system."""
        try:
            # Parse the action request from JSON
            action_request = ActionRequest.parse_raw(action_request_json)
            
            # Get the step from memory
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
            
            # Update the chat history with the action and human feedback
            self._chat_history.extend([
                {"role": "assistant", "content": action_request.action, "name": "GroupChatManager"},
                {"role": "user", "content": f"{step.human_feedback}. Now make the function call", "name": "HumanAgent"},
            ])
            
            try:
                # Execute the appropriate tool based on the action
                # Create a context with all necessary variables
                variables = sk.ContextVariables()
                variables["step_id"] = action_request.step_id
                variables["session_id"] = action_request.session_id
                variables["plan_id"] = action_request.plan_id
                variables["action"] = action_request.action
                variables["chat_history"] = str(self._chat_history)
                
                # Find the appropriate tool to execute
                result = await self._execute_tool_with_llm(variables, context)
                
                # Record the result
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
                
                # Update the step status
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
                
                # Create and return the response
                action_response = ActionResponse(
                    step_id=step.id,
                    plan_id=step.plan_id,
                    session_id=action_request.session_id,
                    result=result,
                    status=StepStatus.completed,
                )
                
                # Publish the message to the group chat manager
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
                
                # Return error response
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

    async def _execute_tool_with_llm(self, variables: sk.ContextVariables, context: Optional[SKContext] = None) -> str:
        """Execute the appropriate tool based on LLM reasoning."""
        # Create a planning context for tool selection
        planner = self._kernel.get_semantic_function("planner", "execute_with_tool")
        
        # Add tool descriptions to the context
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self._tools])
        variables["tools"] = tool_descriptions
        
        # Let the LLM decide which tool to use
        plan_result = await planner.invoke_async(variables)
        tool_name = plan_result.result.strip()
        
        # Find the selected tool
        selected_tool = next((t for t in self._tools if t.name == tool_name), None)
        if not selected_tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # Execute the tool
        tool_result = await selected_tool.invoke_async(variables)
        return tool_result.result

    async def _publish_to_group_chat_manager(self, response: ActionResponse) -> None:
        """Publish a message to the group chat manager."""
        # In Semantic Kernel, we would use events or the connector system
        # This is a simplified implementation
        group_chat_manager_id = f"group_chat_manager_{self._session_id}"
        
        # Create a message connector to send to the group chat manager
        connector = self._kernel.get_connector(group_chat_manager_id)
        if connector:
            await connector.send_async(response.json())
        else:
            logging.warning(f"No connector found for {group_chat_manager_id}")

    def save_state(self) -> Mapping[str, Any]:
        """Save the agent's state."""
        return {"memory": self._memory_store.save_state()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Load the agent's state."""
        self._memory_store.load_state(state["memory"])