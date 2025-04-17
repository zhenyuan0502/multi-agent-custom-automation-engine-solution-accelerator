import pytest
import json
import os

from kernel_agents.agent_base import BaseAgent, DEFAULT_FORMATTING_INSTRUCTIONS
from kernel_agents.agent_factory import AgentFactory
from models.agent_types import AgentType
from kernel_agents.marketing_agent import MarketingAgent

@pytest.mark.asyncio
async def test_dynamic_functions_match_marketing_tools_json():
    # Load marketing tools configuration
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'marketing_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Test dynamic function creation
    for tool in config.get('tools', []):
        name = tool['name']
        template = tool.get('response_template', '')
        fn = BaseAgent.create_dynamic_function(name, template)
        # Prepare dummy args
        kwargs = {}
        for p in tool.get('parameters', []):
            if p['type'] == 'string':
                kwargs[p['name']] = 'test'
            elif p['type'] == 'number':
                kwargs[p['name']] = 1.23
            elif p['type'] == 'array':
                # supply list of correct type
                kwargs[p['name']] = ['test']
        # Invoke function
        result = await fn(**kwargs)
        expected = template.format(**kwargs) + f"\n{DEFAULT_FORMATTING_INSTRUCTIONS}"
        assert result == expected, f"Mismatch for tool {name}: {result} != {expected}"

@pytest.mark.asyncio
async def test_agent_factory_creates_marketing_agent():
    # Use AgentFactory to create MarketingAgent
    agent = await AgentFactory.create_agent(AgentType.MARKETING, 'sessionM', 'userM')
    assert isinstance(agent, MarketingAgent)
    # Validate loaded tools match JSON
    tool_names = [t.name for t in agent._tools]
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'marketing_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    expected = [tool['name'] for tool in config.get('tools', [])]
    assert set(tool_names) == set(expected)

@pytest.mark.asyncio
async def test_llm_agent_has_registered_functions_marketing():
    # Ensure AzureAIAgent fallback has functions registered
    agent = await AgentFactory.create_agent(AgentType.MARKETING, 'sessionY', 'userY')
    azure_agent = agent._agent
    func_store = getattr(azure_agent, '_functions', None) or getattr(azure_agent, 'functions', None)
    assert func_store is not None, "AzureAIAgent missing functions store"
    # Flatten registered names
    if isinstance(func_store, dict):
        names = [fn.name for funcs in func_store.values() for fn in funcs]
    else:
        names = [fn.name for fn in func_store]
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'marketing_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    for name in [tool['name'] for tool in config.get('tools', [])]:
        assert name in names, f"Function {name} not registered in AzureAIAgent"

@pytest.mark.asyncio
async def test_all_marketing_tools_kernels():
    # Load config
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'marketing_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Create agent
    agent = await AgentFactory.create_agent(AgentType.MARKETING, 'sess_allM', 'user_allM')
    for tool in config['tools']:
        name = tool['name']
        fn = next((f for f in agent._tools if f.name == name), None)
        assert fn, f"Tool {name} not loaded"
        # dummy args
        kwargs = {}
        for p in tool.get('parameters', []):
            if p['type'] == 'string':
                kwargs[p['name']] = 'test'
            elif p['type'] == 'number':
                kwargs[p['name']] = 1.23
            elif p['type'] == 'array':
                kwargs[p['name']] = ['test']
        result = await fn.invoke_async(**kwargs)
        assert isinstance(result, str) and result, f"Empty response for {name}"
        for v in kwargs.values():
            assert str(v) in result, f"Value {v} missing in {name} response"
        assert result.strip().endswith(DEFAULT_FORMATTING_INSTRUCTIONS)
