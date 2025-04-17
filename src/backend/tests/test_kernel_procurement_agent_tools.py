import os
import sys
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import semantic_kernel as sk
from context.cosmos_memory_kernel import CosmosMemoryContext
from kernel_agents.procurement_agent import ProcurementAgent
from kernel_agents.agent_base import BaseAgent


def _load_procurement_config():
    tests_dir = os.path.dirname(__file__)
    backend_dir = os.path.dirname(tests_dir)
    json_path = os.path.join(backend_dir, 'tools', 'procurement_tools.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class TestKernelProcurementAgentTools(unittest.TestCase):
    def setUp(self):
        # Load JSON tool names
        config = _load_procurement_config()
        self.expected_names = {tool['name'] for tool in config.get('tools', [])}
        # Create a kernel and memory store
        self.kernel = sk.Kernel()
        self.memory = CosmosMemoryContext(session_id='test', user_id='test')
        # Initialize agent without explicit tools to load from config
        self.agent = ProcurementAgent(
            kernel=self.kernel,
            session_id='test',
            user_id='test',
            memory_store=self.memory
        )

    def test_agent_tools_loaded(self):
        # Ensure the agent's _tools list has KernelFunction objects
        loaded = self.agent._tools
        names = {fn.name for fn in loaded}
        self.assertEqual(
            names,
            self.expected_names,
            f"Loaded tools {names} do not match expected {self.expected_names}"
        )

    def test_dynamic_functions_are_callable(self):
        # For each tool, ensure the dynamic function can be invoked with dummy args
        config = _load_procurement_config()
        for tool in config.get('tools', []):
            fn = next((f for f in self.agent._tools if f.name == tool['name']), None)
            self.assertIsNotNone(fn, f"Function {tool['name']} not found on agent._tools")
            # Build dummy args
            params = tool.get('parameters', [])
            kwargs = {}
            for p in params:
                if p['type'] == 'string':
                    kwargs[p['name']] = 'test'
                elif p['type'] == 'number':
                    kwargs[p['name']] = 1.0
                elif p['type'] == 'array':
                    kwargs[p['name']] = ['a', 'b']
            # Invoke and ensure no exception
            try:
                result = fn.invoke(kwargs) if hasattr(fn, 'invoke') else None
            except Exception as e:
                self.fail(f"Invocation of {tool['name']} raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()