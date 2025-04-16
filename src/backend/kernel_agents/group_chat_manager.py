import logging
import json
from datetime import datetime
import re
from typing import Dict, List, Optional, Any

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    ActionRequest,
    ActionResponse,
    AgentMessage,
    Step,
    StepStatus,
    PlanStatus,
    HumanFeedbackStatus,
)
from event_utils import track_event_if_configured


class GroupChatManager(BaseAgent):
    """Group Chat Manager implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: List[KernelFunction] = None,
        system_message: Optional[str] = None,
        agent_name: str = "GroupChatManager",
        config_path: Optional[str] = None
    ) -> None:
        """Initialize the Group Chat Manager.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "GroupChatManager")
            config_path: Optional path to the group_chat_manager tools configuration file
        """
        # Load configuration if tools not provided
        if tools is None:
            config = self.load_tools_config("group_chat_manager", config_path)
            tools = self.get_tools_from_config(kernel, "group_chat_manager", config_path)
            if not system_message:
                system_message = config.get("system_message", "You are a Group Chat Manager. You coordinate the conversation between different agents and ensure the plan executes smoothly.")
            agent_name = config.get("agent_name", agent_name)
        
        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message
        )
        
        # Dictionary of agent instances for routing
        self._agent_instances = {}

    async def register_agent(self, agent_name: str, agent: BaseAgent) -> None:
        """Register an agent with the Group Chat Manager.
        
        Args:
            agent_name: The name of the agent
            agent: The agent instance
        """
        self._agent_instances[agent_name] = agent
        logging.info(f"Registered agent {agent_name} with Group Chat Manager")

    async def execute_next_step(self, kernel_arguments: KernelArguments) -> str:
        """Execute the next step in the plan.
        
        Args:
            kernel_arguments: Contains session_id and plan_id
            
        Returns:
            Status message
        """
        session_id = kernel_arguments["session_id"]
        plan_id = kernel_arguments["plan_id"]
        
        # Get all steps for the plan
        steps = await self._memory_store.get_steps_for_plan(plan_id, session_id)
        
        # Find the next step to execute (first approved or planned step)
        next_step = None
        for step in steps:
            if step.status == StepStatus.approved or step.status == StepStatus.planned:
                next_step = step
                break
        
        if not next_step:
            # All steps are completed, mark plan as completed
            plan = await self._memory_store.get_plan(plan_id)
            if plan:
                plan.overall_status = PlanStatus.completed
                await self._memory_store.update_plan(plan)
            return "All steps completed. Plan execution finished."
        
        # Update step status to in_progress
        next_step.status = StepStatus.in_progress
        await self._memory_store.update_step(next_step)
        
        # Generate conversation history for context
        plan = await self._memory_store.get_plan(plan_id)
        conversation_history = await self._generate_conversation_history(steps, next_step.id, plan)
        
        # Create action request with conversation history for context
        action_with_history = f"{conversation_history} Here is the step to action: {next_step.action}. ONLY perform the steps and actions required to complete this specific step, the other steps have already been completed. Only use the conversational history for additional information, if it's required to complete the step you have been assigned."
        
        action_request = ActionRequest(
            step_id=next_step.id,
            plan_id=plan_id,
            session_id=session_id,
            action=action_with_history
        )
        
        # Get the appropriate agent
        agent_name = next_step.agent
        if agent_name not in self._agent_instances:
            logging.warning(f"Agent {agent_name} not found. Using GenericAgent instead.")
            agent_name = "GenericAgent"
            if agent_name not in self._agent_instances:
                return f"No agent found to handle step {next_step.id}"
        
        # Log the request
        formatted_agent = re.sub(r"([a-z])([A-Z])", r"\1 \2", agent_name)
        
        # Store the agent message in cosmos
        await self._memory_store.add_item(
            AgentMessage(
                session_id=session_id,
                user_id=self._user_id,
                plan_id=plan_id,
                content=f"Requesting {formatted_agent} to perform action: {next_step.action}",
                source="GroupChatManager",
                step_id=next_step.id,
            )
        )
        
        track_event_if_configured(
            f"Group Chat Manager - Requesting {agent_name} to perform the action and added into the cosmos",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "plan_id": plan_id,
                "content": f"Requesting {agent_name} to perform action: {next_step.action}",
                "source": "GroupChatManager",
                "step_id": next_step.id,
            },
        )
        
        # Special handling for HumanAgent - mark as completed since human feedback is already received
        if agent_name == "HumanAgent":
            # Mark as completed since we have received the human feedback
            next_step.status = StepStatus.completed
            await self._memory_store.update_step(next_step)
            
            logging.info("Marking the step as complete - Since we have received the human feedback")
            track_event_if_configured(
                "Group Chat Manager - Steps completed - Received the human feedback and updated into the cosmos",
                {
                    "session_id": session_id,
                    "user_id": self._user_id,
                    "plan_id": plan_id,
                    "content": "Marking the step as complete - Since we have received the human feedback",
                    "source": agent_name,
                    "step_id": next_step.id,
                },
            )
            return f"Step {next_step.id} for HumanAgent marked as completed"
        else:
            # Send action request to the agent
            agent = self._agent_instances[agent_name]
            await agent.handle_action_request(action_request.json())
            
            return f"Step {next_step.id} execution started with {agent_name}"
            
    async def handle_human_feedback(self, kernel_arguments: KernelArguments) -> str:
        """Handle human feedback on steps.
        
        Args:
            kernel_arguments: Contains human_feedback_json string
            
        Returns:
            Status message
        """
        # Parse the human feedback
        human_feedback_json = kernel_arguments["human_feedback_json"]
        human_feedback = json.loads(human_feedback_json)
        
        session_id = human_feedback.get("session_id", "")
        plan_id = human_feedback.get("plan_id", "")
        step_id = human_feedback.get("step_id", "")
        approved = human_feedback.get("approved", False)
        feedback_text = human_feedback.get("human_feedback", "")
        
        # Get general information
        general_information = f"Today's date is {datetime.now().date()}."
        
        # Get the plan
        plan = await self._memory_store.get_plan(plan_id)
        if not plan:
            return f"Plan {plan_id} not found"
            
        # Get plan human clarification if available
        if hasattr(plan, 'human_clarification_response') and plan.human_clarification_response:
            received_human_feedback_on_plan = (
                plan.human_clarification_response
                + " This information may or may not be relevant to the step you are executing - it was feedback provided by the human user on the overall plan, which includes multiple steps, not just the one you are actioning now."
            )
        else:
            received_human_feedback_on_plan = "No human feedback provided on the overall plan."
            
        # Combine all feedback into a single string
        received_human_feedback = (
            f"{feedback_text} "
            f"{general_information} "
            f"{received_human_feedback_on_plan}"
        )
        
        # Get all steps for the plan
        steps = await self._memory_store.get_steps_for_plan(plan_id, session_id)
        
        # Update and execute the specific step if step_id is provided
        if step_id:
            step = next((s for s in steps if s.id == step_id), None)
            if step:
                await self._update_step_status(step, approved, received_human_feedback)
                if approved:
                    return f"Step {step_id} approved and updated"
                else:
                    # Handle rejected step
                    step.status = StepStatus.rejected
                    if hasattr(step, 'human_approval_status'):
                        step.human_approval_status = HumanFeedbackStatus.rejected
                    await self._memory_store.update_step(step)
                    
                    track_event_if_configured(
                        "Group Chat Manager - Step has been rejected and updated into the cosmos",
                        {
                            "status": StepStatus.rejected,
                            "session_id": session_id,
                            "user_id": self._user_id,
                            "human_approval_status": "rejected",
                            "source": step.agent,
                        },
                    )
                    return f"Step {step_id} rejected"
            else:
                return f"Step {step_id} not found"
        else:
            # Update all steps if no specific step_id is provided
            updates_count = 0
            for step in steps:
                if step.status == StepStatus.planned:
                    await self._update_step_status(step, approved, received_human_feedback)
                    updates_count += 1
            
            return f"Updated {updates_count} steps with human feedback"
            
    async def _update_step_status(self, step: Step, approved: bool, received_human_feedback: str) -> None:
        """Update a step's status based on human feedback.
        
        Args:
            step: The step to update
            approved: Whether the step is approved
            received_human_feedback: Feedback from human
        """
        if approved:
            step.status = StepStatus.approved
            if hasattr(step, 'human_approval_status'):
                step.human_approval_status = HumanFeedbackStatus.accepted
        else:
            step.status = StepStatus.rejected
            if hasattr(step, 'human_approval_status'):
                step.human_approval_status = HumanFeedbackStatus.rejected
                
        step.human_feedback = received_human_feedback
        await self._memory_store.update_step(step)
        
        track_event_if_configured(
            "Group Chat Manager - Received human feedback, Updating step and updated into the cosmos",
            {
                "status": step.status,
                "session_id": step.session_id,
                "user_id": self._user_id,
                "human_feedback": received_human_feedback,
                "source": step.agent,
            },
        )
    
    async def _generate_conversation_history(self, steps: List[Step], current_step_id: str, plan: Any) -> str:
        """Generate conversation history for context.
        
        Args:
            steps: List of all steps
            current_step_id: ID of the current step
            plan: The plan object
            
        Returns:
            Formatted conversation history
        """
        # Initialize the formatted string
        formatted_string = "<conversation_history>Here is the conversation history so far for the current plan. This information may or may not be relevant to the step you have been asked to execute."
        
        # Add plan summary if available
        if hasattr(plan, 'summary') and plan.summary:
            formatted_string += f"The user's task was:\n{plan.summary}\n\n"
        elif hasattr(plan, 'initial_goal') and plan.initial_goal:
            formatted_string += f"The user's task was:\n{plan.initial_goal}\n\n"
            
        formatted_string += "The conversation between the previous agents so far is below:\n"
        
        # Iterate over the steps until the current_step_id
        for i, step in enumerate(steps):
            if step.id == current_step_id:
                break
                
            if step.status == StepStatus.completed and hasattr(step, 'agent_reply') and step.agent_reply:
                formatted_string += f"Step {i}\n"
                formatted_string += f"Group chat manager: {step.action}\n"
                formatted_string += f"{step.agent}: {step.agent_reply}\n"
                
        formatted_string += "</conversation_history>"
        return formatted_string