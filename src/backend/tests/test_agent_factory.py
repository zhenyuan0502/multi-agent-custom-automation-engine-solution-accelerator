import os
import json
import pytest
import semantic_kernel as sk

from kernel_agents.agent_factory import AgentFactory
from models.agent_types import AgentType
from kernel_agents.marketing_agent import MarketingAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from config_kernel import Config

@pytest.mark.asyncio
async def test_agent_factory_creates_marketing_agent_and_registers_functions(monkeypatch):
    # Load JSON config names
    tests_dir = os.path.dirname(__file__)
    backend_dir = os.path.dirname(tests_dir)
    json_path = os.path.join(backend_dir, 'tools', 'marketing_tools.json')
    with open(json_path, 'r') as f:
        config = json.load(f)
    expected_names = {tool['name'] for tool in config.get('tools', [])}

    # Create dummy Azure AI Agent to capture add_function calls
    captured = []
    class DummyAIAgent:
        def __init__(self, *args, **kwargs):
            pass
        def add_function(self, fn):
            captured.append(fn.name)
    
    # Monkeypatch Config.CreateKernel and Config.CreateAzureAIAgent
    dummy_kernel = sk.Kernel()
    monkeypatch.setattr(Config, 'CreateKernel', lambda: dummy_kernel)
    async def dummy_create_azure_ai_agent(kernel, agent_name, instructions):
        return DummyAIAgent()
    monkeypatch.setattr(Config, 'CreateAzureAIAgent', dummy_create_azure_ai_agent)

    # Create agent via factory
    session_id = 'sess'
    user_id = 'user'
    agent = await AgentFactory.create_agent(
        agent_type=AgentType.MARKETING,
        session_id=session_id,
        user_id=user_id
    )

    # Validate agent type
    assert isinstance(agent, MarketingAgent), 'AgentFactory did not create a MarketingAgent'

    # Ensure functions loaded match JSON config
    assert set(captured) == expected_names, \
        f"Registered functions {captured} do not match expected {expected_names}"