import os
import json
import unittest
import asyncio

from kernel_agents.agent_base import BaseAgent, DEFAULT_FORMATTING_INSTRUCTIONS


def _load_marketing_config():
    # Locate the JSON config file relative to this test file
    tests_dir = os.path.dirname(__file__)
    backend_dir = os.path.dirname(tests_dir)
    json_path = os.path.join(backend_dir, 'tools', 'marketing_tools.json')
    with open(json_path, 'r') as f:
        return json.load(f)


def _dummy_value(param):
    # Provide a dummy value based on parameter type
    ptype = param.get('type')
    if ptype == 'string':
        return f"test_{param['name']}"
    if ptype == 'number':
        # Use a float to test formatting
        return 1.23
    if ptype == 'array':
        # Provide a sample list of strings
        return ["val1", "val2"]
    # fallback
    return None


class TestKernelMarketingDynamicFunctions(unittest.TestCase):
    def setUp(self):
        self.config = _load_marketing_config()
        self.tools = self.config.get('tools', [])

    def test_dynamic_functions_formatting(self):
        for tool in self.tools:
            name = tool['name']
            response_template = tool.get('response_template', '')
            # Create the dynamic async function
            dynamic_fn = BaseAgent.create_dynamic_function(name, response_template)
            # Build kwargs for parameters
            params = tool.get('parameters', [])
            kwargs = {p['name']: _dummy_value(p) for p in params}

            # Run the async function
            result = asyncio.run(dynamic_fn(**kwargs))
            # Expected formatted part
            # Format arrays and numbers via Python str for consistency
            expected_core = response_template.format(**kwargs)
            expected = expected_core + f"\n{DEFAULT_FORMATTING_INSTRUCTIONS}"
            self.assertEqual(
                result,
                expected,
                msg=f"Dynamic function '{name}' output mismatch. Expected '{expected}', got '{result}'"
            )

if __name__ == '__main__':
    unittest.main()