import os
import sys
import inspect
import json
import pytest

# allow imports from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.product import get_product_tools


def _load_product_json_config():
    tests_dir = os.path.dirname(__file__)
    backend_dir = os.path.dirname(tests_dir)
    json_path = os.path.join(backend_dir, 'tools', 'product_tools.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_python_and_json_product_tool_names_match():
    python_tools = get_product_tools()
    python_names = {tool.name for tool in python_tools}
    config = _load_product_json_config()
    json_names = {item['name'] for item in config.get('tools', [])}
    assert python_names == json_names, \
        f"Mismatch between Python product tool names and JSON config: {python_names.symmetric_difference(json_names)}"


def test_python_tool_signatures_match_json_parameters():
    config = _load_product_json_config()
    json_tools = {item['name']: item for item in config.get('tools', [])}
    python_tools = get_product_tools()
    for tool in python_tools:
        func = tool.fn
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        json_params = [param['name'] for param in json_tools[tool.name]['parameters']]
        assert param_names == json_params, \
            f"Signature mismatch for '{tool.name}': Python params {param_names}, JSON params {json_params}"