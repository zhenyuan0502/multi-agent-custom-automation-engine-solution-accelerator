import logging
import uuid
from typing import Dict, List, Optional, Annotated

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_function_decorator import kernel_function
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
)


class PlannerAgent(BaseAgent):
    """Planner agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
    ) -> None:
        """Initialize the Planner Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
        """
        super().__init__(
            agent_name="PlannerAgent",
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            agent_type="planner",  # Use agent_type to automatically load tools
            system_message="""
            You are a planner agent. Your role is to create a step-by-step plan to accomplish a user's goal.
            Each step should be clear, actionable, and assigned to the appropriate specialized agent.
            
            When planning:
            1. Break down complex tasks into simpler steps
            2. Consider which agent is best suited for each step
            3. Make sure the steps are in a logical order
            4. Include enough detail for each agent to understand what they need to do
            
            Available agents:
            - HR Agent: For human resources, employee management, benefits, onboarding
            - Marketing Agent: For marketing activities, campaigns, content creation
            - Product Agent: For product information, features, specifications
            - Procurement Agent: For purchasing, supplier management, ordering
            - Tech Support Agent: For technical troubleshooting and IT support
            
            Provide a clear, structured plan that can be executed step by step.
            """
        )
        
        # Register the planning function
        self._register_planning_functions()
    
    def _register_planning_functions(self):
        """Register planning-specific functions with the kernel."""
        # These would be registered automatically through the decorator, but we're being explicit
        functions = [
            self.create_plan,
            self.handle_input_task,
        ]
        for func in functions:
            if hasattr(func, "__kernel_function__"):
                self._kernel.add_function(func)
    
    @kernel_function(
        description="Create a plan based on a user's goal",
        name="create_plan"
    )
    async def create_plan(
        self, 
        context: Annotated[
            KernelArguments, 
            {
                "goal": "The user's goal or task",
                "user_id": "The user's ID",
                "session_id": "The current session ID",
            }
        ]
    ) -> str:
        """Create a detailed plan based on the user's goal."""
        try:
            goal = context["goal"]
            user_id = context.get("user_id", self._user_id)
            session_id = context.get("session_id", self._session_id)
            
            # Add the goal to the chat history
            self._chat_history.append(
                {"role": "user", "content": f"Goal: {goal}"}
            )
            
            # Generate plan steps using the LLM
            planning_prompt = f"""
            Create a detailed step-by-step plan to accomplish this goal: {goal}
            
            For each step, specify:
            1. A descriptive action
            2. Which agent should handle it (HR, Marketing, Product, Procurement, Tech Support, or Generic)
            
            Format your response as a JSON array of steps with 'action' and 'agent' properties.
            Example:
            [
                {{"action": "Research customer demographics", "agent": "Marketing"}},
                {{"action": "Create product specifications document", "agent": "Product"}},
                ...
            ]
            """
            
            # Get the LLM service
            completion_service = self._kernel.get_service("completion")
            
            # Generate the plan steps
            result = await completion_service.complete_chat_async(
                messages=[
                    {"role": "system", "content": self._system_message},
                    {"role": "user", "content": planning_prompt}
                ],
                execution_settings={
                    "response_format": {"type": "json_object"}
                }
            )
            
            # Parse the plan
            import json
            plan_steps = json.loads(result)
            
            # Create a new plan
            plan_id = str(uuid.uuid4())
            plan = Plan(
                id=plan_id,
                session_id=session_id,
                user_id=user_id,
                initial_goal=goal,
                source=self._agent_name,
            )
            
            # Save the plan
            await self._memory_store.add_item(plan)
            
            # Create individual steps
            steps = []
            for i, step_data in enumerate(plan_steps):
                agent_type_str = step_data.get("agent", "Generic")
                try:
                    # Convert string to AgentType enum
                    agent_type = AgentType(agent_type_str.lower())
                except ValueError:
                    # Default to generic agent if the specified agent doesn't exist
                    agent_type = AgentType.generic
                
                step = Step(
                    id=str(uuid.uuid4()),
                    plan_id=plan_id,
                    session_id=session_id,
                    user_id=user_id,
                    action=step_data["action"],
                    agent=agent_type,
                    status=StepStatus.planned,
                )
                steps.append(step)
                
                # Save each step
                await self._memory_store.add_item(step)
            
            # Create plan with steps
            plan_with_steps = PlanWithSteps(
                **plan.model_dump(),
                steps=steps,
                total_steps=len(steps),
                planned=len(steps),
            )
            
            # Return a formatted plan summary
            step_descriptions = "\n".join([
                f"{i+1}. {step.action} (assigned to {step.agent.value} agent)"
                for i, step in enumerate(steps)
            ])
            
            plan_summary = f"""
            Plan created with ID: {plan_id}
            Goal: {goal}
            Number of steps: {len(steps)}
            
            Steps:
            {step_descriptions}
            """
            
            # Log the plan creation
            logging.info(f"Created plan {plan_id} with {len(steps)} steps for goal: {goal}")
            
            # Add the plan to the chat history
            self._chat_history.append(
                {"role": "assistant", "content": plan_summary}
            )
            
            return plan_summary
            
        except Exception as e:
            logging.exception(f"Error creating plan: {e}")
            return f"Error creating plan: {str(e)}"

    @kernel_function(
        description="Handle an input task from the user",
        name="handle_input_task"
    )
    async def handle_input_task(
        self, 
        context: Annotated[
            KernelArguments, 
            {
                "input_task_json": "JSON string of the input task",
            }
        ]
    ) -> str:
        """Handle an input task from the user and create a plan."""
        try:
            input_task_json = context["input_task_json"]
            task = InputTask.parse_raw(input_task_json)
            
            # Create a plan using the goal from the input task
            context["goal"] = task.description
            context["session_id"] = task.session_id
            
            # Create the plan
            return await self.create_plan(context)
            
        except Exception as e:
            logging.exception(f"Error handling input task: {e}")
            return f"Error handling input task: {str(e)}"