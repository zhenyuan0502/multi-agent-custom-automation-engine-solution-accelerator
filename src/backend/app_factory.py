# app_factory.py
import asyncio
import logging
import os
import uuid
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, Query, Request
from middleware.health_check import HealthCheckMiddleware
from auth.auth_utils import get_authenticated_user_details
from config_kernel import Config
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
from utils_kernel import rai_success
from event_utils import track_event_if_configured
from fastapi.middleware.cors import CORSMiddleware
from azure.monitor.opentelemetry import configure_azure_monitor

# Import our new agent factory components
from agents_factory.agent_factory import AgentFactory
from agents_factory.agent_config import AgentBaseConfig
from models.agent_types import AgentType

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


async def get_agents(session_id: str, user_id: str) -> Dict[AgentType, Any]:
    """
    Get or create agent instances for a session, using our new AgentFactory.
    
    Args:
        session_id: The session identifier
        user_id: The user identifier
        
    Returns:
        Dictionary of agent instances by type
    """
    # Use our new AgentFactory to create all agents for this session
    return await AgentFactory.create_all_agents(session_id, user_id)


@app.post("/input_task")
async def input_task_endpoint(input_task: InputTask, request: Request):
    """
    Receive the initial input task from the user.

    ---
    tags:
      - Input Task
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
    planner_agent = agents[AgentType.PLANNER]
    
    # Use the planner to handle the task
    from semantic_kernel.kernel_arguments import KernelArguments
    result = await planner_agent.handle_input_task(
        KernelArguments(input_task_json=input_task.json())
    )
    
    # Get the plan created by the planner
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
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
        
    # Get the agents for this session
    agents = await get_agents(human_feedback.session_id, user_id)
    
    # Send the feedback to the human agent
    human_agent = agents[AgentType.HUMAN]
    
    # Convert feedback to JSON for the kernel function
    from semantic_kernel.kernel_arguments import KernelArguments
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
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
        
    # Get the agents for this session
    agents = await get_agents(human_clarification.session_id, user_id)
    
    # Send the clarification to the planner agent
    planner_agent = agents[AgentType.PLANNER]
    
    # Store the clarification in the plan
    from semantic_kernel.kernel_arguments import KernelArguments
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
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
        
    # Get the agents for this session
    agents = await get_agents(human_feedback.session_id, user_id)
    
    # Handle the approval
    from semantic_kernel.kernel_arguments import KernelArguments
    human_feedback_json = human_feedback.json()
    
    # First process with HumanAgent to update step status
    human_agent = agents[AgentType.HUMAN]
    await human_agent.handle_human_feedback(
        KernelArguments(human_feedback_json=human_feedback_json)
    )
    
    # Then execute the next step with GroupChatManager
    group_chat_manager = agents[AgentType.GROUP_CHAT_MANAGER]
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
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")

    # Initialize memory context
    memory_store = AgentBaseConfig.create_memory_store(session_id or "", user_id)

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
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
    
    # Initialize memory context
    memory_store = await AgentBaseConfig.create_memory_store("", user_id)
    steps = await memory_store.get_steps_for_plan(plan_id=plan_id)
    return steps


@app.get("/agent_messages/{session_id}", response_model=List[AgentMessage])
async def get_agent_messages(session_id: str, request: Request) -> List[AgentMessage]:
    """
    Retrieve agent messages for a specific session.

    ---
    tags:
      - Agent Messages
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        track_event_if_configured("UserIdNotFound", {"status_code": 400, "detail": "no user"})
        raise HTTPException(status_code=400, detail="no user")
    
    # Initialize memory context
    memory_store = await AgentBaseConfig.create_memory_store(session_id, user_id)
    agent_messages = await memory_store.get_data_by_type("agent_message")
    return agent_messages


@app.delete("/messages")
async def delete_all_messages(request: Request) -> Dict[str, str]:
    """
    Delete all messages across sessions.

    ---
    tags:
      - Messages
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        raise HTTPException(status_code=400, detail="no user")
    
    # Initialize memory context
    memory_store = await AgentBaseConfig.create_memory_store("", user_id)
    
    logging.info("Deleting all plans")
    await memory_store.delete_all_items("plan")
    logging.info("Deleting all sessions")
    await memory_store.delete_all_items("session")
    logging.info("Deleting all steps")
    await memory_store.delete_all_items("step")
    logging.info("Deleting all agent_messages")
    await memory_store.delete_all_items("agent_message")
    
    # Clear the agent instances cache
    AgentFactory.clear_cache()
    
    return {"status": "All messages deleted"}


@app.get("/messages")
async def get_all_messages(request: Request):
    """
    Retrieve all messages across sessions.

    ---
    tags:
      - Messages
    """
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]
    if not user_id:
        raise HTTPException(status_code=400, detail="no user")
    
    # Initialize memory context
    memory_store = await AgentBaseConfig.create_memory_store("", user_id)
    message_list = await memory_store.get_all_items()
    return message_list


@app.get("/api/agent-tools")
async def get_agent_tools():
    """
    Retrieve all available agent tools.

    ---
    tags:
      - Agent Tools
    """
    tools_info = []
    
    # Get all agent types
    for agent_type in AgentType:
        # Skip agents that don't have tools
        if agent_type in [AgentType.HUMAN, AgentType.PLANNER, AgentType.GROUP_CHAT_MANAGER]:
            continue
            
        # Get the tool getter for this agent type
        if agent_type in AgentFactory._tool_getters:
            # Create a temporary kernel to get the tools
            kernel = AgentBaseConfig.create_kernel()
            tools = AgentFactory._tool_getters[agent_type](kernel)
            
            # Add tool information
            for tool in tools:
                tools_info.append({
                    "agent": agent_type.value,
                    "function": tool.name,
                    "description": tool.description,
                    "arguments": str(tool.metadata.get("parameters", {}))
                })
    
    return tools_info


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_factory:app", host="127.0.0.1", port=8000, reload=True)