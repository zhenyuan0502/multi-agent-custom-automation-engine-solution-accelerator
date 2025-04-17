import pytest
import json
import os

import semantic_kernel as sk
from kernel_agents.agent_base import DEFAULT_FORMATTING_INSTRUCTIONS
from kernel_agents.agent_factory import AgentFactory
from models.agent_types import AgentType


@pytest.mark.asyncio
async def test_kernel_hr_agent_loads_all_tools():
    # Create HR agent via factory
    agent = await AgentFactory.create_agent(AgentType.HR, 'sess1', 'user1')
    # Tools loaded on the agent
    loaded = [fn.name for fn in agent._tools]
    # Load names from JSON
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools', 'hr_tools.json'))
    with open(config_path, 'r') as f:
        config = json.load(f)
    expected = [tool['name'] for tool in config['tools']]
    assert set(loaded) == set(expected), f"Loaded tools {loaded}, expected {expected}"


@pytest.mark.asyncio
async def test_schedule_orientation_session_kernel_function():
    # Verify that the kernel function produces correct output
    agent = await AgentFactory.create_agent(AgentType.HR, 'sess2', 'user2')
    # Find the function
    fn = next((f for f in agent._tools if f.name == 'schedule_orientation_session'), None)
    assert fn is not None, "schedule_orientation_session not loaded"
    # Invoke function
    result = await fn.invoke_async(employee_name='Alice', date='2025-04-17')
    # Check content and formatting instructions
    assert 'Orientation Session Scheduled' in result
    assert 'Alice' in result and '2025-04-17' in result
    assert result.strip().endswith(DEFAULT_FORMATTING_INSTRUCTIONS)


@pytest.mark.asyncio
async def test_update_policies_kernel_function():
    agent = await AgentFactory.create_agent(AgentType.HR, 'sess3', 'user3')
    fn = next((f for f in agent._tools if f.name == 'update_policies'), None)
    assert fn
    # Invoke with sample data
    result = await fn.invoke_async(policy_name='Dress Code', policy_content='Business casual required')
    assert 'Policy Updated' in result
    assert 'Dress Code' in result
    assert 'Business casual required' in result
    assert result.strip().endswith(DEFAULT_FORMATTING_INSTRUCTIONS)


@pytest.mark.asyncio
async def test_get_hr_information_kernel_function():
    agent = await AgentFactory.create_agent(AgentType.HR, 'sess4', 'user4')
    fn = next((f for f in agent._tools if f.name == 'get_hr_information'), None)
    assert fn
    # Query parameter
    result = await fn.invoke_async(query='onboarding process')
    assert 'HR Information' in result
    # No formatting instruction appended to JSON response_template
    assert result.strip().endswith(DEFAULT_FORMATTING_INSTRUCTIONS)
