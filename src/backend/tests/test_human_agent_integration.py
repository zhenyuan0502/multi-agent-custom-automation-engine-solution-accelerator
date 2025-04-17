import pytest
import json
import os

from kernel_agents.agent_base import BaseAgent, DEFAULT_FORMATTING_INSTRUCTIONS
from kernel_agents.agent_factory import AgentFactory
from models.agent_types import AgentType
from kernel_agents.human_agent import HumanAgent

@pytest.mark.asyncio
async def test_dynamic_functions_match_human_tools_json():
    # Load human tools configuration
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'human_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Test dynamic function creation for each tool
    for tool in config.get('tools', []):
        name = tool['name']
        template = tool.get('response_template') or tool.get('prompt_template', '')
        fn = BaseAgent.create_dynamic_function(name, template)
        # Prepare dummy args
        kwargs = {}
        for p in tool.get('parameters', []):
            if p['type'] == 'string':
                kwargs[p['name']] = 'test'
            elif p['type'] == 'number':
                kwargs[p['name']] = 1.23
        # Invoke function
        result = await fn(**kwargs)
        expected = template.format(**{k: v for k, v in kwargs.items()}) + f"\n{DEFAULT_FORMATTING_INSTRUCTIONS}"
        assert result == expected, f"Mismatch for tool {name}: {result} != {expected}"

@pytest.mark.asyncio
async def test_agent_factory_creates_human_agent():
    # Use AgentFactory to create HumanAgent
    agent = await AgentFactory.create_agent(AgentType.HUMAN, 'sessionX', 'userX')
    assert isinstance(agent, HumanAgent)
    # Validate loaded tools
    tool_names = [t.name for t in agent._tools]
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'human_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    expected = [tool['name'] for tool in config.get('tools', [])]
    assert set(tool_names) == set(expected)

@pytest.mark.asyncio
async def test_llm_agent_has_registered_functions_human():
    # Ensure AzureAIAgent fallback has functions registered
    agent = await AgentFactory.create_agent(AgentType.HUMAN, 'sessionY', 'userY')
    azure_agent = agent._agent
    # Check functions store attribute
    func_store = getattr(azure_agent, '_functions', None) or getattr(azure_agent, 'functions', None)
    assert func_store is not None, "AzureAIAgent missing functions store"
    # Flatten names
    if isinstance(func_store, dict):
        names = [fn.name for funcs in func_store.values() for fn in funcs]
    else:
        names = [fn.name for fn in func_store]
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'human_tools.json')
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    expected = [tool['name'] for tool in config.get('tools', [])]
    for name in expected:
        assert name in names, f"Function {name} not registered in AzureAIAgent"