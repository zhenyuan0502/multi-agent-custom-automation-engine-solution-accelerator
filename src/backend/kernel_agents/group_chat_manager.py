import logging
import json
from typing import Dict, List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel.kernel_arguments import KernelArguments

from multi_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    ActionRequest,
    ActionResponse,
    AgentType,
    Step,
    StepStatus,
)


class GroupChatManager(BaseAgent):
    """Group Chat Manager implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        agents: Dict[str, BaseAgent],
    ) -> None:
        """Initialize the Group Chat Manager.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            agents: Dictionary of available agents by name
        """
        super().__init__(
            agent_name="GroupChatManager",
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=[],  # Group chat manager doesn't need tools
            system_message="""
            You are a group chat manager agent. Your role is to coordinate the execution of a plan by assigning tasks to the appropriate specialized agents and gathering their responses.
            
            Your responsibilities include:
            1. Tracking which steps of the plan are completed and which are still pending
            2. Sending action requests to the appropriate agents
            3. Receiving and processing action responses from agents
            4. Ensuring the plan execution proceeds in the correct order
            5. Handling any issues or errors that occur during plan execution
            
            Available agents:
            - HR Agent: For human resources, employee management, benefits, onboarding
            - Marketing Agent: For marketing activities, campaigns, content creation
            - Product Agent: For product information, features, specifications
            - Procurement Agent: For purchasing, supplier management, ordering
            - Tech Support Agent: For technical troubleshooting and IT support
            - Generic Agent: For general tasks that don't fit with other specialized agents
            """
        )
        
        self._agents = agents
        self._register_group_chat_functions()
    
    def _register_group_chat_functions(self):
        """Register group chat manager specific functions with the kernel."""
        # These would be registered automatically through the decorator, but we're being explicit
        functions = [
            self.handle_action_response,
            self.execute_next_step,
            self.get_next_step,
        ]
        for func in functions:
            if hasattr(func, "__kernel_function__"):
                self._kernel.add_function(func)
    
    @kernel_function(
        description="Handle a response from an agent after performing an action",
        name="handle_action_response"
    )
    @kernel_function_context_parameter(
        name="action_response_json",
        description="JSON string of the action response",
    )
    async def handle_action_response(
        self, context: KernelArguments
    ) -> str:
        """Handle a response from an agent after performing an action."""
        try:
            action_response_json = context["action_response_json"]
            response = ActionResponse.parse_raw(action_response_json)
            
            # Get the step from memory
            step: Step = await self._memory_store.get_step(
                response.step_id, response.session_id
            )
            
            if not step:
                error_message = f"Step {response.step_id} not found in session {response.session_id}"
                logging.error(error_message)
                return error_message
            
            # Update the step status
            step.status = response.status
            if response.result:
                step.agent_reply = response.result
            
            # Save the updated step
            await self._memory_store.update_step(step)
            
            # Log the action response
            logging.info(f"Received action response for step {step.id}. Status: {response.status}")
            
            # Add to chat history
            self._chat_history.append(
                {"role": "assistant", "content": f"Step {step.id} completed with status: {response.status.value}", "name": step.agent.value}
            )
            
            if response.result:
                self._chat_history.append(
                    {"role": "assistant", "content": response.result, "name": step.agent.value}
                )
            
            # Check if there are more steps to execute
            next_step = await self.get_next_step(context)
            if next_step:
                return f"Step {step.id} completed. Proceeding with next step."
            else:
                return f"Step {step.id} completed. No more steps to execute."
            
        except Exception as e:
            logging.exception(f"Error processing action response: {e}")
            return f"Error processing action response: {str(e)}"
    
    @kernel_function(
        description="Execute the next step in the plan",
        name="execute_next_step"
    )
    @kernel_function_context_parameter(
        name="session_id",
        description="The session ID",
    )
    @kernel_function_context_parameter(
        name="plan_id",
        description="The plan ID",
    )
    async def execute_next_step(
        self, context: KernelArguments
    ) -> str:
        """Execute the next step in the plan."""
        try:
            session_id = context.get("session_id", self._session_id)
            plan_id = context["plan_id"]
            
            # Get the next step to execute
            next_step_result = await self.get_next_step(context)
            if not next_step_result:
                return "No more steps to execute."
            
            # Parse the result to get the step
            step_data = json.loads(next_step_result)
            step_id = step_data["id"]
            
            # Get the full step from memory
            step: Step = await self._memory_store.get_step(step_id, session_id)
            if not step:
                error_message = f"Step {step_id} not found in session {session_id}"
                logging.error(error_message)
                return error_message
            
            # Update step status
            step.status = StepStatus.action_requested
            await self._memory_store.update_step(step)
            
            # Create action request
            action_request = ActionRequest(
                step_id=step.id,
                plan_id=step.plan_id,
                session_id=step.session_id,
                action=step.action,
                agent=step.agent,
            )
            
            # Determine which agent to send the request to
            agent_name = f"{step.agent.value}Agent"
            if agent_name.lower() not in [name.lower() for name in self._agents]:
                # Default to generic agent if specified agent doesn't exist
                agent_name = "GenericAgent"
            
            # Find the agent (case insensitive match)
            target_agent = None
            for name, agent in self._agents.items():
                if name.lower() == agent_name.lower():
                    target_agent = agent
                    break
            
            if not target_agent:
                error_message = f"Agent {agent_name} not found"
                logging.error(error_message)
                return error_message
            
            # Send the action request to the agent
            # We're using the agent's handle_action_request function directly
            action_request_json = action_request.json()
            context = KernelArguments(action_request_json=action_request_json)
            
            # Call the agent's handle_action_request function
            result = await target_agent.handle_action_request(context)
            
            logging.info(f"Sent action request for step {step.id} to agent {agent_name}")
            
            return f"Executing step {step.id} with agent {agent_name}: {step.action}"
            
        except Exception as e:
            logging.exception(f"Error executing next step: {e}")
            return f"Error executing next step: {str(e)}"
    
    @kernel_function(
        description="Get the next step to execute from the plan",
        name="get_next_step"
    )
    @kernel_function_context_parameter(
        name="session_id",
        description="The session ID",
    )
    @kernel_function_context_parameter(
        name="plan_id",
        description="The plan ID",
    )
    async def get_next_step(
        self, context: KernelArguments
    ) -> Optional[str]:
        """Get the next step to execute from the plan."""
        try:
            session_id = context.get("session_id", self._session_id)
            plan_id = context["plan_id"]
            
            # Get all steps for the plan
            steps = await self._memory_store.get_steps_for_plan(plan_id, session_id)
            if not steps:
                return None
            
            # Find the next step to execute (first approved or planned step)
            executable_statuses = [StepStatus.approved, StepStatus.planned]
            for step in steps:
                if step.status in executable_statuses:
                    # Return the step as JSON
                    return json.dumps(
                        {
                            "id": step.id,
                            "action": step.action,
                            "agent": step.agent.value,
                        }
                    )
            
            return None
            
        except Exception as e:
            logging.exception(f"Error getting next step: {e}")
            return None