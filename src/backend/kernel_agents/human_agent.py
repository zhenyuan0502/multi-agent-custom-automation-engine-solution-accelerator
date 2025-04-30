import logging
from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    AgentType,
    ApprovalRequest,
    HumanClarification,
    HumanFeedback,
    Step,
    StepStatus,
    AgentMessage,
    ActionRequest,
)
from event_utils import track_event_if_configured
from src.backend.kernel_tools.human_tools import HumanTools


class HumanAgent(BaseAgent):
    """Human agent implementation using Semantic Kernel.

    This agent specializes in representing and assisting humans in the multi-agent system.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.HUMAN.value,
        config_path: Optional[str] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Human Agent.

        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "HumanAgent")
            config_path: Optional path to the Human tools configuration file
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if tools is None:
            # Get tools directly from HumanTools class
            tools_dict = HumanTools.get_all_kernel_functions()
            tools = [KernelFunction.from_method(func) for func in tools_dict.values()]

            # Load the human tools configuration for system message
            config = self.load_tools_config("human", config_path)

            # Use system message from config if not explicitly provided
            if not system_message:
                system_message = config.get(
                    "system_message",
                    "You are an AI Agent. You represent a human employee in our organization. "
                    "You can perform day-to-day activities and collaborate with other functional agents. "
                    "When someone asks you to complete a task, summarize what was done.",
                )

            # Use agent name from config if available
            agent_name = AgentType.HUMAN.value

        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            client=client,
            definition=definition,
        )

    @property
    def plugins(self):
        """Get the plugins for the human agent."""
        return HumanTools.get_all_kernel_functions()

    async def handle_human_feedback(self, human_feedback: HumanFeedback) -> str:
        """Handle human feedback on a step.

        This method processes feedback provided by a human user on a specific step in a plan.
        It updates the step with the feedback, marks the step as completed, and notifies the
        GroupChatManager by creating an ApprovalRequest in the memory store.

        Args:
            human_feedback: The HumanFeedback object containing feedback details
                           including step_id, session_id, and human_feedback text

        Returns:
            Status message indicating success or failure of processing the feedback
        """

        # Get the step
        step = await self._memory_store.get_step(
            human_feedback.step_id, human_feedback.session_id
        )
        if not step:
            return f"Step {human_feedback.step_id} not found"

        # Update the step with the feedback
        step.human_feedback = human_feedback.human_feedback
        step.status = StepStatus.completed

        # Save the updated step
        await self._memory_store.update_step(step)
        await self._memory_store.add_item(
            AgentMessage(
                session_id=human_feedback.session_id,
                user_id=step.user_id,
                plan_id=step.plan_id,
                content=f"Received feedback for step: {step.action}",
                source=AgentType.HUMAN.value,
                step_id=human_feedback.step_id,
            )
        )

        # Track the event
        track_event_if_configured(
            f"Human Agent - Received feedback for step and added into the cosmos",
            {
                "session_id": human_feedback.session_id,
                "user_id": self._user_id,
                "plan_id": step.plan_id,
                "content": f"Received feedback for step: {step.action}",
                "source": AgentType.HUMAN.value,
                "step_id": human_feedback.step_id,
            },
        )

        # Notify the GroupChatManager that the step has been completed
        await self._memory_store.add_item(
            ApprovalRequest(
                session_id=human_feedback.session_id,
                user_id=self._user_id,
                plan_id=step.plan_id,
                step_id=human_feedback.step_id,
                agent_id=AgentType.GROUP_CHAT_MANAGER.value,
            )
        )

        # Track the approval request event
        track_event_if_configured(
            f"Human Agent - Approval request sent for step and added into the cosmos",
            {
                "session_id": human_feedback.session_id,
                "user_id": self._user_id,
                "plan_id": step.plan_id,
                "step_id": human_feedback.step_id,
                "agent_id": "GroupChatManager",
            },
        )

        return "Human feedback processed successfully"

    async def provide_clarification(
        self, human_clarification: HumanClarification
    ) -> str:
        """Provide clarification on a plan.

        This method stores human clarification information for a plan associated with a session.
        It retrieves the plan from memory, updates it with the clarification text, and records
        the event in telemetry.

        Args:
            human_clarification: The HumanClarification object containing the session_id
                                and clarification_text provided by the human user

        Returns:
            Status message indicating success or failure of adding the clarification
        """
        session_id = human_clarification.session_id
        clarification_text = human_clarification.clarification_text

        # Get the plan associated with this session
        plan = await self._memory_store.get_plan_by_session(session_id)
        if not plan:
            return f"No plan found for session {session_id}"

        # Update the plan with the clarification
        plan.human_clarification_response = clarification_text
        await self._memory_store.update_plan(plan)

        # Track the event
        track_event_if_configured(
            "Human Agent - Provided clarification for plan",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "plan_id": plan.id,
                "clarification": clarification_text,
            },
        )

        return f"Clarification provided for plan {plan.id}"
