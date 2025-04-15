# app_kernel.py
import asyncio
import logging
import os
import uuid
from typing import List, Dict, Optional, Any

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

# Azure monitoring
from azure.monitor.opentelemetry import configure_azure_monitor

# Semantic Kernel imports
import semantic_kernel as sk
from semantic_kernel.kernel_arguments import KernelArguments

# Local imports
from middleware.health_check import HealthCheckMiddleware
from auth.auth_utils import get_authenticated_user_details
from config_kernel import Config
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    HumanFeedback,
    HumanClarification,
    InputTask,
    Plan,
    Step,
    AgentMessage,
    PlanWithSteps,
    ActionRequest,
    ActionResponse,
)
from utils_kernel import initialize_runtime_and_context, get_agents, retrieve_all_agent_tools, rai_success
from event_utils import track_event_if_configured
from models.agent_types import AgentType
from kernel_agents.agent_factory import AgentFactory

# Check if the Application Insights Instrumentation Key is set in the environment variables
instrumentation_key = os.getenv("APPLICATIONINSIGHTS_INSTRUMENTATION_KEY")
if instrumentation_key:
    # Configure Application Insights if the Instrumentation Key is found
    configure_azure_monitor(connection_string=instrumentation_key)
    logging.info("Application Insights configured with the provided Instrumentation Key")
else:
    # Log a warning if the Instrumentation Key is not found
    logging.warning("No Application Insights Instrumentation Key found. Skipping configuration")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress INFO logs from 'azure.core.pipeline.policies.http_logging_policy'
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logging.getLogger("azure.identity.aio._internal").setLevel(logging.WARNING)

# Suppress info logs from OpenTelemetry exporter
logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(
    logging.WARNING
)

# Initialize the FastAPI app
app = FastAPI()

frontend_url = Config.FRONTEND_SITE_NAME

# Add this near the top of your app.py, after initializing the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],  # Add your frontend server URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure health check
app.add_middleware(HealthCheckMiddleware, password="", checks={})
logging.info("Added health check middleware")


@app.post("/input_task")
async def input_task_endpoint(input_task: InputTask, request: Request):
    """
    Receive the initial input task from the user.

    ---
    tags:
      - Input Task
    parameters:
      - name: user_principal_id
        in: header
        type: string
        required: true
        description: User ID extracted from the authentication header
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            session_id:
              type: string
              description: Optional session ID, generated if not provided
            description:
              type: string
              description: The task description
            user_id:
              type: string
              description: The user ID associated with the task
    responses:
      200:
        description: Task created successfully
        schema:
          type: object
          properties:
            status:
              type: string
            session_id:
              type: string
            plan_id:
              type: string
            description:
              type: string
            user_id:
              type: string
      400:
        description: Missing or invalid user information
    """

    if not rai_success(input_task.description):
        print("RAI failed")

        track_event_if_configured(
            "RAI failed",
            {
                "status": "Plan not created",
                "description": input_task.description,
                "session_id": input_task.session_id,
            },
        )

        return {
            "status": "Plan not created",
        }
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})

        raise HTTPException(status_code=400, detail="no user")
    if not input_task.session_id:
        input_task.session_id = str(uuid.uuid4())

    # Get the agents for this session
    agents = await get_agents(input_task.session_id, user_id)
    
    # Send the task to the planner agent
    planner_agent = agents["PlannerAgent"]
    
    # Convert input task to JSON for the kernel function
    input_task_json = input_task.json()
    
    # Use the planner to handle the task
    result = await planner_agent.handle_input_task(
        KernelArguments(input_task_json=input_task_json)
    )
    
    # Extract plan ID from the result
    # This is a simplified approach - in a real system, 
    # we would properly parse the result to get the plan ID
    memory_store = planner_agent._memory_store
    plan = await memory_store.get_plan_by_session(input_task.session_id)
    
    if not plan or not plan.id:
        track_event_if_configured(
            "PlanCreationFailed", 
            {
                "session_id": input_task.session_id,
                "description": input_task.description,
            }
        )
        return {
            "status": "Error: Failed to create plan",
            "session_id": input_task.session_id,
            "plan_id": "",
            "description": input_task.description,
        }
    
    # Log custom event for successful input task processing
    track_event_if_configured(
        "InputTaskProcessed",
        {
            "status": f"Plan created with ID: {plan.id}",
            "session_id": input_task.session_id,
            "plan_id": plan.id,
            "description": input_task.description,
        },
    )

    return {
        "status": f"Plan created with ID: {plan.id}",
        "session_id": input_task.session_id,
        "plan_id": plan.id,
        "description": input_task.description,
    }


@app.post("/human_feedback")
async def human_feedback_endpoint(human_feedback: HumanFeedback, request: Request):
    """
    Receive human feedback on a step.

    ---
    tags:
      - Feedback
    parameters:
      - name: user_principal_id
        in: header
        type: string
        required: true
        description: User ID extracted from the authentication header
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            step_id:
              type: string
              description: The ID of the step to provide feedback for
            plan_id:
              type: string
              description: The plan ID
            session_id:
              type: string
              description: The session ID
            approved:
              type: boolean
              description: Whether the step is approved
            human_feedback:
              type: string
              description: Optional feedback details
            updated_action:
              type: string
              description: Optional updated action
            user_id:
              type: string
              description: The user ID providing the feedback
    responses:
      200:
        description: Feedback received successfully
        schema:
          type: object
          properties:
            status:
              type: string
            session_id:
              type: string
            step_id:
              type: string
      400:
        description: Missing or invalid user information
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
        
    # Get the agents for this session
    agents = await get_agents(human_feedback.session_id, user_id)
    
    # Send the feedback to the human agent
    human_agent = agents["HumanAgent"]
    
    # Convert feedback to JSON for the kernel function
    human_feedback_json = human_feedback.json()
    
    # Use the human agent to handle the feedback
    await human_agent.handle_human_feedback(
        KernelArguments(human_feedback_json=human_feedback_json)
    )

    track_event_if_configured(
        "Completed Feedback received",
        {
            "status": "Feedback received",
            "session_id": human_feedback.session_id,
            "step_id": human_feedback.step_id,
        },
    )

    return {
        "status": "Feedback received",
        "session_id": human_feedback.session_id,
        "step_id": human_feedback.step_id,
    }


@app.post("/human_clarification_on_plan")
async def human_clarification_endpoint(
    human_clarification: HumanClarification, request: Request
):
    """
    Receive human clarification on a plan.

    ---
    tags:
      - Clarification
    parameters:
      - name: user_principal_id
        in: header
        type: string
        required: true
        description: User ID extracted from the authentication header
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            plan_id:
              type: string
              description: The plan ID requiring clarification
            session_id:
              type: string
              description: The session ID
            human_clarification:
              type: string
              description: Clarification details provided by the user
            user_id:
              type: string
              description: The user ID providing the clarification
    responses:
      200:
        description: Clarification received successfully
        schema:
          type: object
          properties:
            status:
              type: string
            session_id:
              type: string
      400:
        description: Missing or invalid user information
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
        
    # Get the agents for this session
    agents = await get_agents(human_clarification.session_id, user_id)
    
    # Send the clarification to the planner agent
    planner_agent = agents["PlannerAgent"]
    
    # Convert clarification to kernel arguments
    # For now, we're using a simple approach - in a real system, 
    # the PlannerAgent would have a specific method to handle clarifications
    kernel_args = KernelArguments(
        plan_id=human_clarification.plan_id,
        session_id=human_clarification.session_id,
        human_clarification=human_clarification.human_clarification
    )
    
    # Store the clarification in the plan
    memory_store = planner_agent._memory_store
    plan = await memory_store.get_plan(human_clarification.plan_id)
    if plan:
        plan.human_clarification_request = human_clarification.human_clarification
        await memory_store.update_plan(plan)

    track_event_if_configured(
        "Completed Human clarification on the plan",
        {
            "status": "Clarification received",
            "session_id": human_clarification.session_id,
        },
    )

    return {
        "status": "Clarification received",
        "session_id": human_clarification.session_id,
    }


@app.post("/approve_step_or_steps")
async def approve_step_endpoint(
    human_feedback: HumanFeedback, request: Request
) -> Dict[str, str]:
    """
    Approve a step or multiple steps in a plan.

    ---
    tags:
      - Approval
    parameters:
      - name: user_principal_id
        in: header
        type: string
        required: true
        description: User ID extracted from the authentication header
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            step_id:
              type: string
              description: Optional step ID to approve
            plan_id:
              type: string
              description: The plan ID
            session_id:
              type: string
              description: The session ID
            approved:
              type: boolean
              description: Whether the step(s) are approved
            human_feedback:
              type: string
              description: Optional feedback details
            updated_action:
              type: string
              description: Optional updated action
            user_id:
              type: string
              description: The user ID providing the approval
    responses:
      200:
        description: Approval status returned
        schema:
          type: object
          properties:
            status:
              type: string
      400:
        description: Missing or invalid user information
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
        
    # Get the agents for this session
    agents = await get_agents(human_feedback.session_id, user_id)
    
    # Send the approval to the group chat manager
    group_chat_manager = agents["GroupChatManager"]
    
    # Handle the approval
    human_feedback_json = human_feedback.json()
    
    # First process with HumanAgent to update step status
    human_agent = agents["HumanAgent"]
    await human_agent.handle_human_feedback(
        KernelArguments(human_feedback_json=human_feedback_json)
    )
    
    # Then execute the next step with GroupChatManager
    await group_chat_manager.execute_next_step(
        KernelArguments(
            session_id=human_feedback.session_id,
            plan_id=human_feedback.plan_id
        )
    )

    # Return a status message
    if human_feedback.step_id:
        track_event_if_configured(
            "Completed Human clarification with step_id",
            {
                "status": f"Step {human_feedback.step_id} - Approval:{human_feedback.approved}."
            },
        )

        return {
            "status": f"Step {human_feedback.step_id} - Approval:{human_feedback.approved}."
        }
    else:
        track_event_if_configured(
            "Completed Human clarification without step_id",
            {"status": "All steps approved"},
        )

        return {"status": "All steps approved"}


@app.get("/plans", response_model=List[PlanWithSteps])
async def get_plans(
    request: Request, session_id: Optional[str] = Query(None)
) -> List[PlanWithSteps]:
    """
    Retrieve plans for the current user.

    ---
    tags:
      - Plans
    parameters:
      - name: session_id
        in: query
        type: string
        required: false
        description: Optional session ID to retrieve plans for a specific session
    responses:
      200:
        description: List of plans with steps for the user
        schema:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
                description: Unique ID of the plan
              session_id:
                type: string
                description: Session ID associated with the plan
              initial_goal:
                type: string
                description: The initial goal derived from the user's input
              overall_status:
                type: string
                description: Status of the plan (e.g., in_progress, completed)
              steps:
                type: array
                items:
                  type: object
                  properties:
                    id:
                      type: string
                      description: Unique ID of the step
                    plan_id:
                      type: string
                      description: ID of the plan the step belongs to
                    action:
                      type: string
                      description: The action to be performed
                    agent:
                      type: string
                      description: The agent responsible for the step
                    status:
                      type: string
                      description: Status of the step (e.g., planned, approved, completed)
      400:
        description: Missing or invalid user information
      404:
        description: Plan not found
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")

    # Initialize memory context
    memory_store = CosmosMemoryContext(session_id or "", user_id)

    if session_id:
        plan = await memory_store.get_plan_by_session(session_id=session_id)
        if not plan:
            track_event_if_configured(
                "GetPlanBySessionNotFound",
                {"status_code": 400, "detail": "Plan not found"},
            )
            raise HTTPException(status_code=404, detail="Plan not found")

        steps = await memory_store.get_steps_for_plan(plan.id, session_id)
        plan_with_steps = PlanWithSteps(**plan.model_dump(), steps=steps)
        plan_with_steps.update_step_counts()
        return [plan_with_steps]

    all_plans = await memory_store.get_all_plans()
    # Fetch steps for all plans concurrently
    steps_for_all_plans = await asyncio.gather(
        *[memory_store.get_steps_for_plan(plan.id, plan.session_id) for plan in all_plans]
    )
    # Create list of PlanWithSteps and update step counts
    list_of_plans_with_steps = []
    for plan, steps in zip(all_plans, steps_for_all_plans):
        plan_with_steps = PlanWithSteps(**plan.model_dump(), steps=steps)
        plan_with_steps.update_step_counts()
        list_of_plans_with_steps.append(plan_with_steps)

    return list_of_plans_with_steps


@app.get("/steps/{plan_id}", response_model=List[Step])
async def get_steps_by_plan(plan_id: str, request: Request) -> List[Step]:
    """
    Retrieve steps for a specific plan.

    ---
    tags:
      - Steps
    parameters:
      - name: plan_id
        in: path
        type: string
        required: true
        description: The ID of the plan to retrieve steps for
    responses:
      200:
        description: List of steps associated with the specified plan
        schema:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
                description: Unique ID of the step
              plan_id:
                type: string
                description: ID of the plan the step belongs to
              action:
                type: string
                description: The action to be performed
              agent:
                type: string
                description: The agent responsible for the step
              status:
                type: string
                description: Status of the step (e.g., planned, approved, completed)
              agent_reply:
                type: string
                description: Optional response from the agent after execution
              human_feedback:
                type: string
                description: Optional feedback provided by a human
              updated_action:
                type: string
                description: Optional modified action based on feedback
      400:
        description: Missing or invalid user information
      404:
        description: Plan or steps not found
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
    
    # Initialize memory context
    memory_store = CosmosMemoryContext("", user_id)
    steps = await memory_store.get_steps_for_plan(plan_id=plan_id)
    return steps


@app.get("/agent_messages/{session_id}", response_model=List[AgentMessage])
async def get_agent_messages(session_id: str, request: Request) -> List[AgentMessage]:
    """
    Retrieve agent messages for a specific session.

    ---
    tags:
      - Agent Messages
    parameters:
      - name: session_id
        in: path
        type: string
        required: true
        description: The ID of the session to retrieve agent messages for
    responses:
      200:
        description: List of agent messages associated with the specified session
        schema:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
                description: Unique ID of the agent message
              session_id:
                type: string
                description: Session ID associated with the message
              plan_id:
                type: string
                description: Plan ID related to the agent message
              content:
                type: string
                description: Content of the message
              source:
                type: string
                description: Source of the message (e.g., agent type)
              timestamp:
                type: string
                format: date-time
                description: Timestamp of the message
              step_id:
                type: string
                description: Optional step ID associated with the message
      400:
        description: Missing or invalid user information
      404:
        description: Agent messages not found
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
    
    # Initialize memory context
    memory_store = CosmosMemoryContext(session_id, user_id)
    agent_messages = await memory_store.get_data_by_type("agent_message")
    return agent_messages


@app.delete("/messages")
async def delete_all_messages(request: Request) -> Dict[str, str]:
    """
    Delete all messages across sessions.

    ---
    tags:
      - Messages
    responses:
      200:
        description: Confirmation of deletion
        schema:
          type: object
          properties:
            status:
              type: string
              description: Status message indicating all messages were deleted
      400:
        description: Missing or invalid user information
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        raise HTTPException(status_code=400, detail="no user")
    
    # Initialize memory context
    memory_store = CosmosMemoryContext(session_id="", user_id=user_id)
    
    logging.info("Deleting all plans")
    await memory_store.delete_all_items("plan")
    logging.info("Deleting all sessions")
    await memory_store.delete_all_items("session")
    logging.info("Deleting all steps")
    await memory_store.delete_all_items("step")
    logging.info("Deleting all agent_messages")
    await memory_store.delete_all_items("agent_message")
    
    # Clear the agent factory cache
    AgentFactory.clear_cache()
    
    return {"status": "All messages deleted"}


@app.get("/messages")
async def get_all_messages(request: Request):
    """
    Retrieve all messages across sessions.

    ---
    tags:
      - Messages
    responses:
      200:
        description: List of all messages across sessions
        schema:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
                description: Unique ID of the message
              data_type:
                type: string
                description: Type of the message (e.g., session, step, plan, agent_message)
              session_id:
                type: string
                description: Session ID associated with the message
              user_id:
                type: string
                description: User ID associated with the message
              content:
                type: string
                description: Content of the message
              timestamp:
                type: string
                format: date-time
                description: Timestamp of the message
      400:
        description: Missing or invalid user information
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        raise HTTPException(status_code=400, detail="no user")
    
    # Initialize memory context
    memory_store = CosmosMemoryContext(session_id="", user_id=user_id)
    message_list = await memory_store.get_all_items()
    return message_list


@app.get("/api/agent-tools")
async def get_agent_tools():
    """
    Retrieve all available agent tools.

    ---
    tags:
      - Agent Tools
    responses:
      200:
        description: List of all available agent tools and their descriptions
        schema:
          type: array
          items:
            type: object
            properties:
              agent:
                type: string
                description: Name of the agent associated with the tool
              function:
                type: string
                description: Name of the tool function
              description:
                type: string
                description: Detailed description of what the tool does
              arguments:
                type: string
                description: Arguments required by the tool function
    """
    return retrieve_all_agent_tools()


# Initialize the application when it starts
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup.
    
    This function runs when the FastAPI application starts up.
    It sets up the agent types and tool loaders so the first request is faster.
    """
    # Log startup
    logging.info("Application starting up. Initializing agent factory...")
    
    try:
        # Create a temporary session and user ID to pre-initialize agents
        # This ensures tools are loaded into the factory on startup
        temp_session_id = "startup-session"
        temp_user_id = "startup-user"
        
        # Create a test agent to initialize the tool loading system
        # This will pre-load tool configurations into memory
        test_agent = await AgentFactory.create_agent(
            agent_type=AgentType.GENERIC,
            session_id=temp_session_id,
            user_id=temp_user_id
        )
        
        # Clean up initialization resources
        AgentFactory.clear_cache(temp_session_id)
        logging.info("Agent factory successfully initialized")
        
    except Exception as e:
        logging.error(f"Error initializing agent factory: {e}")
        # Don't fail startup, but log the error


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_kernel:app", host="127.0.0.1", port=8000, reload=True)