import os
import sys
import json
import pytest
import semantic_kernel as sk

# allow imports from backend directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernel_agents.procurement_agent import ProcurementAgent
from kernel_agents.agent_factory import AgentFactory
from models.agent_types import AgentType
from context.cosmos_memory_kernel import CosmosMemoryContext
from config_kernel import Config


def _load_procurement_json_config():
    tests_dir = os.path.dirname(__file__)
    backend_dir = os.path.dirname(tests_dir)
    json_path = os.path.join(backend_dir, 'tools', 'procurement_tools.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_procurement_agent_loads_all_tools_from_config():
    kernel = sk.Kernel()
    memory = CosmosMemoryContext(session_id='test', user_id='test')
    # Instantiate without explicit tools
    agent = ProcurementAgent(kernel=kernel, session_id='test', user_id='test', memory_store=memory)
    loaded_names = {fn.name for fn in agent._tools}
    config = _load_procurement_json_config()
    expected_names = {tool['name'] for tool in config.get('tools', [])}
    assert loaded_names == expected_names, \
        f"ProcurementAgent loaded {loaded_names}, expected {expected_names}"

@pytest.mark.asyncio
async def test_agent_factory_creates_procurement_agent_and_registers_functions(monkeypatch):
    config = _load_procurement_json_config()
    expected_names = {tool['name'] for tool in config.get('tools', [])}
    captured = []
    class DummyAIAgent:
        def __init__(self, *args, **kwargs): pass
        def add_function(self, fn): captured.append(fn.name)
    # Monkeypatch kernel and AzureAIAgent creation
    dummy_kernel = sk.Kernel()
    monkeypatch.setattr(Config, 'CreateKernel', lambda: dummy_kernel)
    async def dummy_create_azure_ai_agent(*args, **kwargs):
        return DummyAIAgent()
    monkeypatch.setattr(Config, 'CreateAzureAIAgent', dummy_create_azure_ai_agent)

    # Create via factory
    agent = await AgentFactory.create_agent(
        agent_type=AgentType.PROCUREMENT,
        session_id='sess',
        user_id='user'
    )
    assert isinstance(agent, ProcurementAgent), 'AgentFactory did not create a ProcurementAgent'
    assert set(captured) == expected_names, \
        f"Registered functions {captured} do not match expected procurement tools {expected_names}"