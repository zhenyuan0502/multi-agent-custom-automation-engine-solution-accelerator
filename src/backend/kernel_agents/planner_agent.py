import logging
import uuid
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    AgentMessage,
    InputTask,
    Plan,
    Step,
    StepStatus,
    PlanStatus,
    HumanFeedbackStatus,
)
from event_utils import track_event_if_configured
from app_config import config

# Define structured output models
class StructuredOutputStep(BaseModel):
    action: str = Field(description="Detailed description of the step action")
    agent: str = Field(description="Name of the agent to execute this step")

class StructuredOutputPlan(BaseModel):
    initial_goal: str = Field(description="The goal of the plan")
    steps: List[StructuredOutputStep] = Field(description="List of steps to achieve the goal")
    summary_plan_and_steps: str = Field(description="Brief summary of the plan and steps")
    human_clarification_request: Optional[str] = Field(None, description="Any additional information needed from the human")

class PlannerAgent(BaseAgent):
    """Planner agent implementation using Semantic Kernel.
    
    This agent creates and manages plans based on user tasks, breaking them down into steps
    that can be executed by specialized agents to achieve the user's goal.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = "PlannerAgent",
        config_path: Optional[str] = None,
        available_agents: List[str] = None,
        agent_tools_list: List[str] = None,
        agent_instances: Optional[Dict[str, BaseAgent]] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Planner Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: Optional list of tools for this agent
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "PlannerAgent")
            config_path: Optional path to the configuration file
            available_agents: List of available agent names for creating steps
            agent_tools_list: List of available tools across all agents
            agent_instances: Dictionary of agent instances available to the planner
            client: Optional client instance (passed to BaseAgent)
            definition: Optional definition instance (passed to BaseAgent)
        """
        # Default system message if not provided
        if not system_message:
            system_message = "You are a Planner agent responsible for creating and managing plans. You analyze tasks, break them down into steps, and assign them to the appropriate specialized agents."
        
        # Initialize the base agent
        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            agent_type="planner",  # Use planner_tools.json if available
            client=client,
            definition=definition
        )
        
        # Store additional planner-specific attributes
        self._available_agents = available_agents or ["HumanAgent", "HrAgent", "MarketingAgent", 
                                                     "ProductAgent", "ProcurementAgent", 
                                                     "TechSupportAgent", "GenericAgent"]
        self._agent_tools_list = agent_tools_list or []
        self._agent_instances = agent_instances or {}
        
        # Create the Azure AI Agent for planning operations
        # This will be initialized in async_init
        self._azure_ai_agent = None
        
    async def async_init(self) -> None:
        """Asynchronously initialize the PlannerAgent.
        
        Creates the Azure AI Agent for planning operations.
        
        Returns:
            None
        """
        try:
            # Create the Azure AI Agent using AppConfig
            self._azure_ai_agent = await config.create_azure_ai_agent(
                kernel=self._kernel,
                agent_name="PlannerAgent",
                instructions=self._generate_instruction(""),
                temperature=0.0
            )
            logging.info("Successfully created Azure AI Agent for PlannerAgent")
            return True
        except Exception as e:
            logging.error(f"Failed to create Azure AI Agent for PlannerAgent: {e}")
            raise
        
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
        
        # Generate a structured plan with steps
        plan, steps = await self._create_structured_plan(input_task)

        print(f"Plan created: {plan}")

        print(f"Steps created: {steps}")

        
        if steps:
            # Add a message about the created plan
            await self._memory_store.add_item(
                AgentMessage(
                    session_id=input_task.session_id,
                    user_id=self._user_id,
                    plan_id=plan.id,
                    content=f"Generated a plan with {len(steps)} steps. Click the blue check box beside each step to complete it, click the x to remove this step.",
                    source="PlannerAgent",
                    step_id="",
                )
            )
            
            track_event_if_configured(
                f"Planner - Generated a plan with {len(steps)} steps and added plan into the cosmos",
                {
                    "session_id": input_task.session_id,
                    "user_id": self._user_id,
                    "plan_id": plan.id,
                    "content": f"Generated a plan with {len(steps)} steps. Click the blue check box beside each step to complete it, click the x to remove this step.",
                    "source": "PlannerAgent",
                },
            )
            
            # If human clarification is needed, add a message requesting it
            if hasattr(plan, 'human_clarification_request') and plan.human_clarification_request:
                await self._memory_store.add_item(
                    AgentMessage(
                        session_id=input_task.session_id,
                        user_id=self._user_id,
                        plan_id=plan.id,
                        content=f"I require additional information before we can proceed: {plan.human_clarification_request}",
                        source="PlannerAgent",
                        step_id="",
                    )
                )
                
                track_event_if_configured(
                    "Planner - Additional information requested and added into the cosmos",
                    {
                        "session_id": input_task.session_id,
                        "user_id": self._user_id,
                        "plan_id": plan.id,
                        "content": f"I require additional information before we can proceed: {plan.human_clarification_request}",
                        "source": "PlannerAgent",
                    },
                )
        
        return f"Plan '{plan.id}' created successfully with {len(steps)} steps"
        
    async def handle_plan_clarification(self, kernel_arguments: KernelArguments) -> str:
        """Handle human clarification for a plan.
        
        Args:
            kernel_arguments: Contains session_id and human_clarification
            
        Returns:
            Status message
        """
        session_id = kernel_arguments["session_id"]
        human_clarification = kernel_arguments["human_clarification"]
        
        # Retrieve and update the plan
        plan = await self._memory_store.get_plan_by_session(session_id)
        if not plan:
            return f"No plan found for session {session_id}"
            
        plan.human_clarification_response = human_clarification
        await self._memory_store.update_plan(plan)
        
        # Add a record of the clarification
        await self._memory_store.add_item(
            AgentMessage(
                session_id=session_id,
                user_id=self._user_id,
                plan_id="",
                content=f"{human_clarification}",
                source="HumanAgent",
                step_id="",
            )
        )
        
        track_event_if_configured(
            "Planner - Store HumanAgent clarification and added into the cosmos",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "content": f"{human_clarification}",
                "source": "HumanAgent",
            },
        )
        
        # Add a confirmation message
        await self._memory_store.add_item(
            AgentMessage(
                session_id=session_id,
                user_id=self._user_id,
                plan_id="",
                content="Thanks. The plan has been updated.",
                source="PlannerAgent",
                step_id="",
            )
        )
        
        track_event_if_configured(
            "Planner - Updated with HumanClarification and added into the cosmos",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "content": "Thanks. The plan has been updated.",
                "source": "PlannerAgent",
            },
        )
        
        return "Plan updated with human clarification"
        
    async def _create_structured_plan(self, input_task: InputTask) -> Tuple[Plan, List[Step]]:
        """Create a structured plan with steps based on the input task.
        
        Args:
            input_task: The input task from the user
            
        Returns:
            Tuple containing the created plan and list of steps
        """
        try:
            # Generate the instruction for the LLM
            instruction = self._generate_instruction(input_task.description)
            
            # Log the input task for debugging
            logging.info(f"Creating plan for task: '{input_task.description}'")
            logging.info(f"Using available agents: {self._available_agents}")
            
            # Use the Azure AI Agent instead of direct function invocation
            if self._azure_ai_agent is None:
                # Initialize the agent if it's not already done
                await self.async_init()
                
            if self._azure_ai_agent is None:
                raise RuntimeError("Failed to initialize Azure AI Agent for planning")
                
            # Log detailed information about the instruction being sent
            logging.info(f"Invoking PlannerAgent with instruction length: {len(instruction)}")
            
            # Create kernel arguments - make sure we explicitly emphasize the task
            kernel_args = KernelArguments()
            kernel_args["input"] = f"TASK: {input_task.description}\n\n{instruction}"
            
            print(f"Kernel arguments: {kernel_args}")

            # Call invoke with proper keyword arguments
            response_content = ""
            
            # Use keyword arguments instead of positional arguments
            # Set a lower temperature to ensure consistent results
            async_generator = self._azure_ai_agent.invoke(
                arguments=kernel_args,
                settings={
                    "temperature": 0.0
                }
            )
            
            # Collect the response from the async generator
            async for chunk in async_generator:
                if chunk is not None:
                    response_content += str(chunk)
            
            print(f"Response content: {response_content}")
            
            # Debug the response
            logging.info(f"Response content length: {len(response_content)}")
            logging.debug(f"Response content first 500 chars: {response_content[:500]}")
            # Log more of the response for debugging
            logging.info(f"Full response: {response_content}")
            
            # Check if response is empty or whitespace
            if not response_content or response_content.isspace():
                raise ValueError("Received empty response from Azure AI Agent")
            
            # Parse the JSON response using the structured output model
            try:
                # First try to parse using Pydantic model
                try:
                    parsed_result = StructuredOutputPlan.parse_raw(response_content)
                except Exception as e1:
                    logging.warning(f"Failed to parse direct JSON with Pydantic: {str(e1)}")
                    
                    # If direct parsing fails, try to extract JSON first
                    json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                        logging.info(f"Found JSON content in markdown code block, length: {len(json_content)}")
                        try:
                            parsed_result = StructuredOutputPlan.parse_raw(json_content)
                        except Exception as e2:
                            logging.warning(f"Failed to parse extracted JSON with Pydantic: {str(e2)}")
                            # Try conventional JSON parsing as fallback
                            json_data = json.loads(json_content)
                            parsed_result = StructuredOutputPlan.parse_obj(json_data)
                    else:
                        # Try to extract JSON without code blocks - maybe it's embedded in text
                        # Look for patterns like { ... } that contain "initial_goal" and "steps"
                        json_pattern = r'\{.*?"initial_goal".*?"steps".*?\}'
                        alt_match = re.search(json_pattern, response_content, re.DOTALL)
                        
                        if alt_match:
                            potential_json = alt_match.group(0)
                            logging.info(f"Found potential JSON in text, length: {len(potential_json)}")
                            try:
                                json_data = json.loads(potential_json)
                                parsed_result = StructuredOutputPlan.parse_obj(json_data)
                            except Exception as e3:
                                logging.warning(f"Failed to parse potential JSON: {str(e3)}")
                                # If all extraction attempts fail, try parsing the whole response as JSON
                                json_data = json.loads(response_content)
                                parsed_result = StructuredOutputPlan.parse_obj(json_data)
                        else:
                            # If we can't find JSON patterns, create a fallback plan from the text
                            logging.info("Using fallback plan creation from text response")
                            return await self._create_fallback_plan_from_text(input_task, response_content)
                
                # Extract plan details and log for debugging
                initial_goal = parsed_result.initial_goal
                steps_data = parsed_result.steps
                summary = parsed_result.summary_plan_and_steps
                human_clarification_request = parsed_result.human_clarification_request
                
                # Log potential mismatches between task and plan for debugging
                if "onboard" in input_task.description.lower() and "marketing" in initial_goal.lower():
                    logging.warning(f"Potential mismatch: Task was about onboarding but plan goal mentions marketing: {initial_goal}")
                    
                # Log the steps and agent assignments for debugging
                for i, step in enumerate(steps_data):
                    logging.info(f"Step {i+1} - Agent: {step.agent}, Action: {step.action}")
                
                # Create the Plan instance
                plan = Plan(
                    id=str(uuid.uuid4()),
                    session_id=input_task.session_id,
                    user_id=self._user_id,
                    initial_goal=initial_goal,
                    overall_status=PlanStatus.in_progress,
                    summary=summary,
                    human_clarification_request=human_clarification_request
                )
                
                # Store the plan
                await self._memory_store.add_plan(plan)
                
                track_event_if_configured(
                    "Planner - Initial plan and added into the cosmos",
                    {
                        "session_id": input_task.session_id,
                        "user_id": self._user_id,
                        "initial_goal": initial_goal,
                        "overall_status": PlanStatus.in_progress,
                        "source": "PlannerAgent",
                        "summary": summary,
                        "human_clarification_request": human_clarification_request,
                    },
                )
                
                # Create steps from the parsed data
                steps = []
                for step_data in steps_data:
                    action = step_data.action
                    agent_name = step_data.agent
                    
                    # Log any unusual agent assignments for debugging
                    if "onboard" in input_task.description.lower() and agent_name != "HrAgent":
                        logging.warning(f"UNUSUAL AGENT ASSIGNMENT: Task contains 'onboard' but assigned to {agent_name} instead of HrAgent")
                    
                    # Validate agent name
                    if agent_name not in self._available_agents:
                        logging.warning(f"Invalid agent name: {agent_name}, defaulting to GenericAgent")
                        agent_name = "GenericAgent"
                    
                    # Create the step
                    step = Step(
                        id=str(uuid.uuid4()),
                        plan_id=plan.id,
                        session_id=input_task.session_id,
                        user_id=self._user_id,
                        action=action,
                        agent=agent_name,
                        status=StepStatus.planned,
                        human_approval_status=HumanFeedbackStatus.requested
                    )
                    
                    # Store the step
                    await self._memory_store.add_step(step)
                    steps.append(step)
                    
                    track_event_if_configured(
                        "Planner - Added planned individual step into the cosmos",
                        {
                            "plan_id": plan.id,
                            "action": action,
                            "agent": agent_name,
                            "status": StepStatus.planned,
                            "session_id": input_task.session_id,
                            "user_id": self._user_id,
                            "human_approval_status": HumanFeedbackStatus.requested,
                        },
                    )
                
                return plan, steps
                
            except Exception as e:
                # If JSON parsing fails, log error and create error plan
                logging.exception(f"Failed to parse JSON response: {e}")
                logging.info(f"Raw response was: {response_content[:1000]}...")
                # Try a fallback approach
                return await self._create_fallback_plan_from_text(input_task, response_content)
                
        except Exception as e:
            logging.exception(f"Error creating structured plan: {e}")
            
            track_event_if_configured(
                f"Planner - Error in create_structured_plan: {e} into the cosmos",
                {
                    "session_id": input_task.session_id,
                    "user_id": self._user_id,
                    "initial_goal": "Error generating plan",
                    "overall_status": PlanStatus.failed,
                    "source": "PlannerAgent",
                    "summary": f"Error generating plan: {e}",
                },
            )
            
            # Create an error plan
            error_plan = Plan(
                id=str(uuid.uuid4()),
                session_id=input_task.session_id,
                user_id=self._user_id,
                initial_goal="Error generating plan",
                overall_status=PlanStatus.failed,
                summary=f"Error generating plan: {str(e)}"
            )
            
            await self._memory_store.add_plan(error_plan)
            return error_plan, []
    
    async def _create_fallback_plan_from_text(self, input_task: InputTask, text_content: str) -> Tuple[Plan, List[Step]]:
        """Create a plan from unstructured text when JSON parsing fails.
        
        Args:
            input_task: The input task
            text_content: The text content from the LLM
            
        Returns:
            Tuple containing the created plan and list of steps
        """
        logging.info("Creating fallback plan from text content")
        
        # Extract goal from the text (first line or use input task description)
        goal_match = re.search(r"(?:Goal|Initial Goal|Plan):\s*(.+?)(?:\n|$)", text_content)
        goal = goal_match.group(1).strip() if goal_match else input_task.description
        
        # Create the plan
        plan = Plan(
            id=str(uuid.uuid4()),
            session_id=input_task.session_id,
            user_id=self._user_id,
            initial_goal=goal,
            overall_status=PlanStatus.in_progress,
            summary=f"Plan created from {input_task.description}"
        )
        
        # Store the plan
        await self._memory_store.add_plan(plan)
        
        # Parse steps using regex
        step_pattern = re.compile(r'(?:Step|)\s*(\d+)[:.]\s*\*?\*?(?:Agent|):\s*\*?([^:*\n]+)\*?[:\s]*(.+?)(?=(?:Step|)\s*\d+[:.]\s*|$)', re.DOTALL)
        matches = step_pattern.findall(text_content)
        
        if not matches:
            # Fallback to simpler pattern
            step_pattern = re.compile(r'(\d+)[.:\)]\s*([^:]*?):\s*(.*?)(?=\d+[.:\)]|$)', re.DOTALL)
            matches = step_pattern.findall(text_content)
        
        # If still no matches, look for bullet points or numbered lists
        if not matches:
            step_pattern = re.compile(r'[•\-*]\s*([^:]*?):\s*(.*?)(?=[•\-*]|$)', re.DOTALL)
            bullet_matches = step_pattern.findall(text_content)
            if bullet_matches:
                # Convert bullet matches to our expected format (number, agent, action)
                matches = []
                for i, (agent_text, action) in enumerate(bullet_matches, 1):
                    matches.append((str(i), agent_text.strip(), action.strip()))
        
        steps = []
        # If we found no steps at all, create at least one generic step
        if not matches:
            generic_step = Step(
                id=str(uuid.uuid4()),
                plan_id=plan.id,
                session_id=input_task.session_id,
                user_id=self._user_id,
                action=f"Process the request: {input_task.description}",
                agent="GenericAgent",
                status=StepStatus.planned,
                human_approval_status=HumanFeedbackStatus.requested
            )
            await self._memory_store.add_step(generic_step)
            steps.append(generic_step)
        else:
            for match in matches:
                number = match[0].strip()
                agent_text = match[1].strip()
                action = match[2].strip()
                
                # Clean up agent name
                agent = re.sub(r'\s+', '', agent_text)
                if not agent or agent not in self._available_agents:
                    agent = "GenericAgent"  # Default to GenericAgent if not recognized
                    
                # Create and store the step
                step = Step(
                    id=str(uuid.uuid4()),
                    plan_id=plan.id,
                    session_id=input_task.session_id,
                    user_id=self._user_id,
                    action=action,
                    agent=agent,
                    status=StepStatus.planned,
                    human_approval_status=HumanFeedbackStatus.requested
                )
                
                await self._memory_store.add_step(step)
                steps.append(step)
                
        return plan, steps
    
    def _generate_instruction(self, objective: str) -> str:
        """Generate instruction for the LLM to create a plan.
        
        Args:
            objective: The user's objective
            
        Returns:
            Instruction string for the LLM
        """
        # Create a list of available agents
        agents_str = ", ".join(self._available_agents)
        
        # Create list of available tools
        # If _agent_tools_list is empty but we have agent instances available elsewhere,
        # we should retrieve tools directly from agent instances
        tools_str = ""
        if hasattr(self, '_agent_instances') and self._agent_instances:
            # Extract tools from agent instances
            agent_tools_sections = []
            
            # Process each agent to get their tools
            for agent_name, agent in self._agent_instances.items():
                if hasattr(agent, '_tools') and agent._tools:
                    # Create a section header for this agent
                    agent_tools_sections.append(f"### {agent_name} Tools ###")
                    
                    # Add each tool from this agent
                    for tool in agent._tools:
                        if hasattr(tool, 'name') and hasattr(tool, 'description'):
                            tool_desc = f"Agent: {agent_name} - Function: {tool.name} - {tool.description}"
                            agent_tools_sections.append(tool_desc)
                    
                    # Add a blank line after each agent's tools
                    agent_tools_sections.append("")
            
            # Join all sections
            if agent_tools_sections:
                tools_str = "\n".join(agent_tools_sections)
                # Log the tools for debugging
                logging.debug(f"Generated tools list from agent instances with {len(agent_tools_sections)} entries")
            else:
                tools_str = "Various specialized tools (No tool details available from agent instances)"
                logging.warning("No tools found in agent instances")
        elif self._agent_tools_list:
            # Fall back to the existing tools list if available
            tools_str = "\n".join(self._agent_tools_list)
            logging.debug(f"Using existing agent_tools_list with {len(self._agent_tools_list)} entries")
        else:
            # Default fallback
            tools_str = "Various specialized tools"
            logging.warning("No tools information available for planner instruction")
        
        # Build the instruction, avoiding backslashes in f-string expressions
        objective_part = f"Your objective is:\n{objective}" if objective else "When given an objective, analyze it and create a plan to accomplish it."
        
        return f"""
        You are the Planner, an AI orchestrator that manages a group of AI agents to accomplish tasks.

        For the given objective, come up with a simple step-by-step plan.
        This plan should involve individual tasks that, if executed correctly, will yield the correct answer. Do not add any superfluous steps.
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        These actions are passed to the specific agent. Make sure the action contains all the information required for the agent to execute the task.
        
        {objective_part}

        The agents you have access to are:
        {agents_str}

        These agents have access to the following functions:
        {tools_str}

        IMPORTANT AGENT SELECTION GUIDANCE:
        - HrAgent: ALWAYS use for ALL employee-related tasks like onboarding, hiring, benefits, payroll, training, employee records, ID cards, mentoring, background checks, etc.
        - MarketingAgent: Use for marketing campaigns, branding, market research, content creation, social media, etc.
        - ProcurementAgent: Use for purchasing, vendor management, supply chain, asset management, etc.
        - ProductAgent: Use for product development, roadmaps, features, product feedback, etc.
        - TechSupportAgent: Use for technical issues, software/hardware setup, troubleshooting, IT support, etc.
        - GenericAgent: Use only for general knowledge tasks that don't fit other categories
        - HumanAgent: Use only when human input is absolutely required and no other agent can handle the task

        The first step of your plan should be to ask the user for any additional information required to progress the rest of steps planned.

        Only use the functions provided as part of your plan. If the task is not possible with the agents and tools provided, create a step with the agent of type Exception and mark the overall status as completed.

        Do not add superfluous steps - only take the most direct path to the solution, with the minimum number of steps. Only do the minimum necessary to complete the goal.

        If there is a single function call that can directly solve the task, only generate a plan with a single step. For example, if someone asks to be granted access to a database, generate a plan with only one step involving the grant_database_access function, with no additional steps.

        You must prioritise using the provided functions to accomplish each step. First evaluate each and every function the agents have access too. Only if you cannot find a function needed to complete the task, and you have reviewed each and every function, and determined why each are not suitable, there are two options you can take when generating the plan.
        First evaluate whether the step could be handled by a typical large language model, without any specialised functions. For example, tasks such as "add 32 to 54", or "convert this SQL code to a python script", or "write a 200 word story about a fictional product strategy".

        If a general Large Language Model CAN handle the step/required action, add a step to the plan with the action you believe would be needed, and add "EXCEPTION: No suitable function found. A generic LLM model is being used for this step." to the end of the action. Assign these steps to the GenericAgent. For example, if the task is to convert the following SQL into python code (SELECT * FROM employees;), and there is no function to convert SQL to python, write a step with the action "convert the following SQL into python code (SELECT * FROM employees;) EXCEPTION: No suitable function found. A generic LLM model is being used for this step." and assign it to the GenericAgent.

        Alternatively, if a general Large Language Model CAN NOT handle the step/required action, add a step to the plan with the action you believe would be needed, and add "EXCEPTION: Human support required to do this step, no suitable function found." to the end of the action. Assign these steps to the HumanAgent. For example, if the task is to find the best way to get from A to B, and there is no function to calculate the best route, write a step with the action "Calculate the best route from A to B. EXCEPTION: Human support required, no suitable function found." and assign it to the HumanAgent.

        Limit the plan to 6 steps or less.

        Choose from {agents_str} ONLY for planning your steps.
        
        When generating the action in the plan, frame the action as an instruction you are passing to the agent to execute. It should be a short, single sentence. Include the function to use. For example, "Set up an Office 365 Account for Jessica Smith. Function: set_up_office_365_account"

        Ensure the summary of the plan and the overall steps is less than 50 words.

        Identify any additional information that might be required to complete the task. Include this information in the plan in the human_clarification_request field of the plan. If it is not required, leave it as null. Do not include information that you are waiting for clarification on in the string of the action field, as this otherwise won't get updated.
        
        Return your response as a JSON object with the following structure:
        {{
          "initial_goal": "The goal of the plan",
          "steps": [
            {{
              "action": "Detailed description of the step action",
              "agent": "AgentName"
            }}
          ],
          "summary_plan_and_steps": "Brief summary of the plan and steps",
          "human_clarification_request": "Any additional information needed from the human" 
        }}
        """