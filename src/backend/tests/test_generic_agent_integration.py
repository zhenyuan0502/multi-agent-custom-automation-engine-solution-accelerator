import sys
import os
import pytest
import asyncio

# Ensure src/backend is on the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernel_agents.agent_factory import AgentFactory
from models.agent_types import AgentType
from config_kernel import Config

@pytest.mark.asyncio
async def test_create_and_execute_generic_agent():
    """
    Integration test: create a GenericAgent from AgentFactory, ensure tools load from generic_tools.json,
    and execute the dummy_function.
    """
    session_id = "integration-test-session"
    user_id = "integration-test-user"

    # Create the agent using the real config and factory
    agent = await AgentFactory.create_agent(
        agent_type=AgentType.GENERIC,
        session_id=session_id,
        user_id=user_id
    )
    assert agent is not None, "AgentFactory did not return an agent"
    
    # Find the dummy_function tool
    dummy_tool = next((t for t in agent._tools if hasattr(t, 'name') and t.name == "dummy_function"), None)
    assert dummy_tool is not None, "dummy_function not loaded in GenericAgent tools"

    # Execute the dummy_function (should not require parameters)
    if hasattr(dummy_tool, 'invoke_async') and asyncio.iscoroutinefunction(dummy_tool.invoke_async):
        result = await dummy_tool.invoke_async()
    elif asyncio.iscoroutinefunction(dummy_tool):
        result = await dummy_tool()
    else:
        result = dummy_tool()

    assert result.strip() == "This is a placeholder function", f"Unexpected result: {result}"
