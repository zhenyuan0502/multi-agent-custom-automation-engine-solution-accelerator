import os
import sys
import json
import unittest
import asyncio

# allow imports from backend directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernel_agents.agent_base import BaseAgent, DEFAULT_FORMATTING_INSTRUCTIONS


def _load_procurement_config():
    # Locate the JSON config file
    tests_dir = os.path.dirname(__file__)
    backend_dir = os.path.dirname(tests_dir)
    json_path = os.path.join(backend_dir, 'tools', 'procurement_tools.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _dummy_value(param):
    ptype = param.get('type')
    if ptype == 'string':
        return f"test_{param['name']}"
    if ptype == 'number':
        # use float for number
        return 3.14
    if ptype == 'array':
        return ["val1", "val2"]
    return None


class TestKernelProcurementDynamicFunctions(unittest.TestCase):
    def setUp(self):
        self.config = _load_procurement_config()
        self.tools = self.config.get('tools', [])

    def test_dynamic_functions_formatting(self):
        for tool in self.tools:
            name = tool['name']
            template = tool.get('response_template', '')
            # Create the dynamic async function
            dynamic_fn = BaseAgent.create_dynamic_function(name, template)
            # Prepare kwargs based on parameters
            params = tool.get('parameters', [])
            kwargs = {p['name']: _dummy_value(p) for p in params}

            # Invoke the async function
            result = asyncio.run(dynamic_fn(**kwargs))
            # Build expected response
            expected_core = template.format(**kwargs)
            expected = expected_core + f"\n{DEFAULT_FORMATTING_INSTRUCTIONS}"
            self.assertEqual(
                result,
                expected,
                msg=f"Dynamic function '{name}' output mismatch. Expected '{expected}', got '{result}'"
            )

if __name__ == '__main__':
    unittest.main()