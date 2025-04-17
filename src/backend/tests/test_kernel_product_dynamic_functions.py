import os
import sys
import json
import unittest
import asyncio

# allow imports from backend directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernel_agents.agent_base import BaseAgent, DEFAULT_FORMATTING_INSTRUCTIONS


def _load_product_config():
    tests_dir = os.path.dirname(__file__)
    backend_dir = os.path.dirname(tests_dir)
    json_path = os.path.join(backend_dir, 'tools', 'product_tools.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _dummy_value(param):
    ptype = param.get('type')
    if ptype == 'string':
        return f"test_{param['name']}"
    if ptype == 'number':
        return 2.5
    if ptype == 'integer':
        return 3
    if ptype == 'array':
        return ['a', 'b']
    return None


class TestKernelProductDynamicFunctions(unittest.TestCase):
    def setUp(self):
        self.config = _load_product_config()
        self.tools = self.config.get('tools', [])

    def test_dynamic_functions_formatting(self):
        for tool in self.tools:
            name = tool['name']
            template = tool.get('response_template', '')
            # Create dynamic function
            dynamic_fn = BaseAgent.create_dynamic_function(name, template)
            # Build kwargs
            params = tool.get('parameters', [])
            kwargs = {p['name']: _dummy_value(p) for p in params}
            # Invoke async function
            result = asyncio.run(dynamic_fn(**kwargs))
            # Expected core
            expected_core = template.format(**kwargs)
            expected = expected_core + f"\n{DEFAULT_FORMATTING_INSTRUCTIONS}"
            self.assertEqual(
                result,
                expected,
                msg=f"Dynamic function '{name}' output mismatch. Expected '{expected}', got '{result}'"
            )

if __name__ == '__main__':
    unittest.main()