import logging
from typing import Any, Dict, List, Mapping, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.kernel_arguments import KernelArguments
# Import Azure AI Agent
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent

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
from config_kernel import Config
from event_utils import track_event_if_configured

class BaseAgent:
    """BaseAgent implemented using Semantic Kernel's AzureAIAgent."""

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
        self._agent_name = agent_name
        self._kernel = kernel
        self._session_id = session_id
        self._user_id = user_id
        self._memory_store = memory_store
        self._tools = tools
        self._system_message = system_message
        
        # Create Azure AI Agent instance
        self._agent = Config.CreateAzureAIAgent(
            kernel=self._kernel,
            agent_name=self._agent_name,
            instructions=self._system_message
        )
        
        # Register tools with the agent
        for tool in self._tools:
            self._agent.add_function(tool)
        
        # Register the action handler
        self._register_handler_functions()

    def _register_handler_functions(self):
        """Register this agent's handler functions with the kernel."""
        # Import this agent's handle_action_request method as a kernel function
        self._kernel.add_function(
            self.handle_action_request, 
            plugin_name=self._agent_name, 
            function_name="handle_action_request"
        )

    async def handle_action_request(
        self, action_request_json: str
    ) -> str:
        """Handle an action request from another agent or the system."""
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
            
            try:
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
                result_content = result.value

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
                    "Base agent - Error during agent execution, captured into the cosmos",
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