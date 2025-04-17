import pytest
import json
import os

from kernel_agents.agent_base import BaseAgent, DEFAULT_FORMATTING_INSTRUCTIONS
from kernel_agents.agent_factory import AgentFactory
from models.agent_types import AgentType
from kernel_agents.hr_agent import HrAgent

@pytest.mark.asyncio
async def test_dynamic_functions_match_original_templates():
    # Load HR tools configuration
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'hr_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Test each tool's dynamic function output
    for tool in config.get('tools', []):
        name = tool['name']
        params = tool.get('parameters', [])
        template = tool.get('response-template') or tool.get('response_template', '')
        # Create dynamic function
        fn = BaseAgent.create_dynamic_function(name, template)
        # Prepare dummy arguments
        kwargs = {}
        for p in params:
            if p['type'] == 'string':
                kwargs[p['name']] = 'test'
            elif p['type'] == 'number':
                kwargs[p['name']] = 1.23
        # Invoke function and check output
        result = await fn(**kwargs)
        expected = template.format(**kwargs) + f"\n{DEFAULT_FORMATTING_INSTRUCTIONS}"
        assert result == expected, f"Mismatch for tool {name}: {result} != {expected}"

@pytest.mark.asyncio
async def test_agent_factory_creates_hr_agent():
    # Create an HR agent via the factory
    agent = await AgentFactory.create_agent(AgentType.HR, 'test_session', 'test_user')
    # Validate correct agent class
    assert isinstance(agent, HrAgent)
    # Validate loaded tools match configuration
    tool_names = [t.name for t in agent._tools]
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'hr_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    expected_names = [tool['name'] for tool in config.get('tools', [])]
    assert set(tool_names) == set(expected_names)

@pytest.mark.asyncio
async def test_llm_agent_has_registered_functions():
    # Ensure AzureAIAgent has the functions available for LLM calls
    agent = await AgentFactory.create_agent(AgentType.HR, 'test_session2', 'test_user2')
    azure_agent = agent._agent  # AzureAIAgent instance
    # Determine functions attribute
    function_store = getattr(azure_agent, '_functions', None) or getattr(azure_agent, 'functions', None)
    assert function_store is not None, "AzureAIAgent missing functions store"
    # Flatten function names
    if isinstance(function_store, dict):
        registered_names = [fn.name for funcs in function_store.values() for fn in funcs]
    else:
        registered_names = [fn.name for fn in function_store]
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'hr_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    expected = [tool['name'] for tool in config.get('tools', [])]
    for name in expected:
        assert name in registered_names, f"Function {name} not registered in AzureAIAgent"