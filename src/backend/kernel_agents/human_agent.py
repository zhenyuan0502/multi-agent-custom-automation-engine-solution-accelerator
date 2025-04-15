import logging
from typing import List, Annotated

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    HumanFeedback,
    HumanFeedbackStatus,
    Step,
    StepStatus,
)


class HumanAgent(BaseAgent):
    """Human agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
    ) -> None:
        """Initialize the Human Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
        """
        super().__init__(
            agent_name="HumanAgent",
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=[],  # Human agent doesn't need tools
            system_message="You are a human user. You will be asked for feedback on steps in a plan."
        )

    @kernel_function(
        description="Handle feedback from a human on a planned step",
        name="handle_human_feedback"
    )
    async def handle_human_feedback(
        self, 
        human_feedback_json: Annotated[str, "JSON string containing human feedback on a step"]
    ) -> str:
        """Handle feedback from a human user on a proposed step in the plan."""
        try:
            feedback = HumanFeedback.parse_raw(human_feedback_json)
            
            # Get the step from memory
            step: Step = await self._memory_store.get_step(
                feedback.step_id, feedback.session_id
            )
            
            if step:
                # Update the step based on feedback
                step.human_approval_status = (
                    HumanFeedbackStatus.accepted if feedback.approved 
                    else HumanFeedbackStatus.rejected
                )
                
                if feedback.human_feedback:
                    step.human_feedback = feedback.human_feedback
                
                if feedback.updated_action:
                    step.updated_action = feedback.updated_action
                
                # Update the step status
                if feedback.approved:
                    step.status = StepStatus.approved
                    # Add a message to the chat history
                    self._chat_history.append(
                        {"role": "user", "content": f"I approve this step. {feedback.human_feedback or ''}"}
                    )
                else:
                    step.status = StepStatus.rejected
                    # Add a message to the chat history
                    self._chat_history.append(
                        {"role": "user", "content": f"I reject this step. {feedback.human_feedback or ''}"}
                    )
                
                # Save the updated step
                await self._memory_store.update_step(step)
                
                logging.info(f"Step {step.id} updated with human feedback. Approved: {feedback.approved}")
                
                # Return success message
                return f"Human feedback processed for step {step.id}. Approved: {feedback.approved}"
            else:
                logging.error(f"Step {feedback.step_id} not found in session {feedback.session_id}")
                return f"Error: Step {feedback.step_id} not found"
        
        except Exception as e:
            logging.exception(f"Error processing human feedback: {e}")
            return f"Error processing human feedback: {str(e)}"