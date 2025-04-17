import inspect
import json
import os
import pytest

from agents.marketing import get_marketing_tools
from kernel_agents.marketing_agent import MarketingAgent
import semantic_kernel as sk
from context.cosmos_memory_kernel import CosmosMemoryContext


def _load_marketing_json_config():
    # Path to the marketing_tools.json file in the backend/tools directory
    tests_dir = os.path.dirname(__file__)
    backend_dir = os.path.dirname(tests_dir)
    json_path = os.path.join(backend_dir, 'tools', 'marketing_tools.json')
    with open(json_path, 'r') as f:
        return json.load(f)


def test_python_and_json_tool_names_match():
    # Load tools defined in Python
    python_tools = get_marketing_tools()
    python_names = {tool.name for tool in python_tools}
    # Load JSON configuration
    config = _load_marketing_json_config()
    json_names = {item['name'] for item in config.get('tools', [])}
    assert python_names == json_names, \
        f"Mismatch between Python tool names and JSON config: {python_names.symmetric_difference(json_names)}"


def test_python_tool_signatures_match_json_parameters():
    # Load JSON configuration
    config = _load_marketing_json_config()
    json_tools = {item['name']: item for item in config.get('tools', [])}

    # Inspect each Python tool function signature
    python_tools = get_marketing_tools()
    for tool in python_tools:
        # Get underlying Python function object
        func = tool.fn
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        json_params = [param['name'] for param in json_tools[tool.name]['parameters']]
        assert param_names == json_params, \
            f"Signature mismatch for '{tool.name}': Python params {param_names}, JSON params {json_params}"


def test_marketing_agent_loads_all_tools_from_config():
    # Initialize a kernel and memory context
    kernel = sk.Kernel()
    memory = CosmosMemoryContext(session_id='test', user_id='test')
    # Instantiate the agent using the class directly
    agent = MarketingAgent(kernel=kernel, session_id='test', user_id='test', memory_store=memory)
    # The agent should load tools when constructed without explicit tools
    loaded_funcs = {fn.name for fn in agent._tools}
    # Compare against JSON config names
    config = _load_marketing_json_config()
    json_names = {item['name'] for item in config.get('tools', [])}
    assert loaded_funcs == json_names, \
        f"Agent loaded tools {loaded_funcs} do not match JSON config {json_names}"