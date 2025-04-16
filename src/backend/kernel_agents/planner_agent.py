import logging
import uuid
import json
import re
from typing import Dict, List, Optional, Annotated

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    AgentType,
    InputTask,
    Plan,
    PlanWithSteps,
    Step,
    StepStatus,
    PlanStatus,
)


class PlannerAgent(BaseAgent):
    """Planner agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: List[KernelFunction] = None,
        system_message: Optional[str] = None,
        agent_name: str = "PlannerAgent",
        config_path: Optional[str] = None
    ) -> None:
        """Initialize the Planner Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "PlannerAgent")
            config_path: Optional path to the Planner tools configuration file
        """
        # Load configuration if tools not provided
        if tools is None:
            config = self.load_tools_config("planner", config_path)
            tools = self.get_tools_from_config(kernel, "planner", config_path)
            if not system_message:
                system_message = config.get("system_message", "You are a planner agent. You create and manage action plans to help users achieve their goals.")
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
        
    async def handle_input_task(self, kernel_arguments: KernelArguments) -> str:
        """Handle the initial input task from the user.
        
        Args:
            kernel_arguments: Contains the input_task_json string
            
        Returns:
            Status message
        """
        # Parse the input task
        input_task_json = kernel_arguments["input_task_json"]
        input_task = InputTask.parse_raw(input_task_json)
        
        # Generate a plan
        plan = await self._create_plan(input_task)
        
        # Store the plan
        await self._memory_store.add_plan(plan)
        
        # Generate steps for the plan
        steps = await self._create_steps(plan)
        
        # Store the steps
        for step in steps:
            await self._memory_store.add_step(step)
            
        return f"Plan '{plan.id}' created successfully with {len(steps)} steps"
        
    async def _create_plan(self, input_task: InputTask) -> Plan:
        """Create a plan based on the input task.
        
        Args:
            input_task: The input task
            
        Returns:
            A new plan
        """
        # Generate plan ID
        plan_id = str(uuid.uuid4())
        
        # Ask the LLM to generate a goal based on the input task
        messages = [{
            "role": "user", 
            "content": f"Based on this task description: '{input_task.description}', create a concise goal statement."
        }]
        
        result = await self._agent.invoke_async(messages=messages)
        goal = result.value.strip()
        
        # Create the plan
        return Plan(
            id=plan_id,
            session_id=input_task.session_id,
            user_id=input_task.user_id,
            initial_goal=goal,
            overall_status=PlanStatus.in_progress
        )
        
    async def _create_steps(self, plan: Plan) -> List[Step]:
        """Create steps for the plan.
        
        Args:
            plan: The plan to create steps for
            
        Returns:
            List of steps
        """
        # Ask the LLM to generate steps for the plan
        messages = [{
            "role": "user", 
            "content": f"Create a step-by-step plan to achieve this goal: '{plan.initial_goal}'. For each step, specify which agent should handle it (HrAgent, MarketingAgent, ProductAgent, ProcurementAgent, TechSupportAgent, or GenericAgent) and describe the action in detail."
        }]
        
        result = await self._agent.invoke_async(messages=messages)
        steps_text = result.value.strip()
        
        # Parse the steps from the LLM response
        steps = []
        
        # Use regex to extract steps
        step_pattern = re.compile(r'(\d+)\.\s*\*\*([\w\s]+)\*\*:\s*(.*?)(?=\d+\.\s*\*\*|\Z)', re.DOTALL)
        matches = step_pattern.findall(steps_text)
        
        if not matches:
            # Fallback to simple numbered lines
            step_pattern = re.compile(r'(\d+)\.\s*([\w\s]+):\s*(.*?)(?=\d+\.\s*|\Z)', re.DOTALL)
            matches = step_pattern.findall(steps_text)
            
        if not matches:
            # Second fallback - just create a generic step
            steps.append(
                Step(
                    id=str(uuid.uuid4()),
                    plan_id=plan.id,
                    session_id=plan.session_id,
                    action=f"Review and implement: {steps_text}",
                    agent="GenericAgent",
                    status=StepStatus.planned
                )
            )
            return steps
        
        # Create steps from the parsed text
        for match in matches:
            number = match[0]
            agent = match[1].strip().replace(" ", "")  # Remove spaces in agent name
            action = match[2].strip()
            
            # Create the step
            steps.append(
                Step(
                    id=str(uuid.uuid4()),
                    plan_id=plan.id,
                    session_id=plan.session_id,
                    action=action,
                    agent=agent,
                    status=StepStatus.planned
                )
            )
            
        return steps