import logging
import json
from typing import Dict, List, Optional, Annotated, Any

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    ActionRequest,
    ActionResponse,
    AgentType,
    Step,
    StepStatus,
    PlanStatus,
)


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
        
        # Create action request
        action_request = ActionRequest(
            step_id=next_step.id,
            plan_id=plan_id,
            session_id=session_id,
            action=next_step.action
        )
        
        # Get the appropriate agent
        agent_name = next_step.agent
        if agent_name not in self._agent_instances:
            logging.warning(f"Agent {agent_name} not found. Using GenericAgent instead.")
            agent_name = "GenericAgent"
            if agent_name not in self._agent_instances:
                return f"No agent found to handle step {next_step.id}"
        
        # Send action request to the agent
        agent = self._agent_instances[agent_name]
        await agent.handle_action_request(action_request.json())
        
        return f"Step {next_step.id} execution started with {agent_name}"