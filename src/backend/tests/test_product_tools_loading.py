import os
import sys
import json
import pytest
import semantic_kernel as sk

# allow imports from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernel_agents.product_agent import ProductAgent
from kernel_agents.agent_factory import AgentFactory
from models.agent_types import AgentType
from context.cosmos_memory_kernel import CosmosMemoryContext
from config_kernel import Config


def _load_product_json_config():
    tests_dir = os.path.dirname(__file__)
    backend_dir = os.path.dirname(tests_dir)
    json_path = os.path.join(backend_dir, 'tools', 'product_tools.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_product_agent_loads_all_tools_from_config():
    kernel = sk.Kernel()
    memory = CosmosMemoryContext(session_id='sid', user_id='uid')
    # Instantiate without explicit tools
    agent = ProductAgent(kernel=kernel, session_id='sid', user_id='uid', memory_store=memory)
    loaded = {fn.name for fn in agent._tools}
    config = _load_product_json_config()
    expected = {tool['name'] for tool in config.get('tools', [])}
    assert loaded == expected, f"ProductAgent loaded {loaded}, expected {expected}"

@pytest.mark.asyncio
async def test_agent_factory_creates_product_agent_and_registers_functions(monkeypatch):
    config = _load_product_json_config()
    expected = {tool['name'] for tool in config.get('tools', [])}
    captured = []
    class DummyAIAgent:
        def __init__(self, *args, **kwargs): pass
        def add_function(self, fn): captured.append(fn.name)
    # Monkeypatch CreateKernel and CreateAzureAIAgent
    dummy_kernel = sk.Kernel()
    monkeypatch.setattr(Config, 'CreateKernel', lambda: dummy_kernel)
    async def dummy_create_azure_ai_agent(*args, **kwargs):
        return DummyAIAgent()
    monkeypatch.setattr(Config, 'CreateAzureAIAgent', dummy_create_azure_ai_agent)

    # Create via factory
    agent = await AgentFactory.create_agent(
        agent_type=AgentType.PRODUCT,
        session_id='sess',
        user_id='user'
    )
    from kernel_agents.product_agent import ProductAgent as PA
    assert isinstance(agent, PA), 'AgentFactory did not create a ProductAgent'
    assert set(captured) == expected, f"Registered functions {captured} do not match expected {expected}"