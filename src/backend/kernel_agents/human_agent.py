import logging
from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import HumanFeedback, Step, StepStatus

class HumanAgent(BaseAgent):
    """Human agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: List[KernelFunction] = None,
        system_message: Optional[str] = None,
        agent_name: str = "HumanAgent",
        config_path: Optional[str] = None
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
        """
        # Load configuration if tools not provided
        if tools is None:
            config = self.load_tools_config("human", config_path)
            tools = self.get_tools_from_config(kernel, "human", config_path)
            if not system_message:
                system_message = config.get("system_message", "You represent a human user in the system. You provide feedback and clarifications to help the AI agents better serve the user.")
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
        
    async def handle_human_feedback(self, kernel_arguments: KernelArguments) -> str:
        """Handle human feedback on a step.
        
        Args:
            kernel_arguments: Contains the human_feedback_json string
            
        Returns:
            Status message
        """
        # Parse the human feedback
        human_feedback_json = kernel_arguments["human_feedback_json"]
        human_feedback = HumanFeedback.parse_raw(human_feedback_json)
        
        # Get the step
        step = await self._memory_store.get_step(human_feedback.step_id, human_feedback.session_id)
        if not step:
            return f"Step {human_feedback.step_id} not found"
        
        # Update the step with the feedback
        step.human_feedback = human_feedback.human_feedback
        step.updated_action = human_feedback.updated_action
        
        if human_feedback.approved:
            step.status = StepStatus.approved
        else:
            step.status = StepStatus.needs_update
        
        # Save the updated step
        await self._memory_store.update_step(step)
        
        # If approved and updated action is provided, update the step's action
        if human_feedback.approved and human_feedback.updated_action:
            step.action = human_feedback.updated_action
            await self._memory_store.update_step(step)
        
        return "Human feedback processed successfully"