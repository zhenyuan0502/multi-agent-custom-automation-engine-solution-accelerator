import inspect
import time
import logging
from datetime import datetime
from typing import Annotated, Callable, List

from semantic_kernel.functions import kernel_function


class GenericTools:
    """Define Generic Agent functions (tools)"""

    @staticmethod
    @kernel_function(
        description="This is a placeholder function, for a proper Azure AI Search RAG process."
    )
    async def dummy_function() -> str:
        # This is a placeholder function, for a proper Azure AI Search RAG process.

        """This is a placeholder"""
        return "This is a placeholder function"

    @staticmethod
    def get_all_kernel_functions() -> dict[str, Callable]:
        """
        Returns a dictionary of all methods in this class that have the @kernel_function annotation.
        This function itself is not annotated with @kernel_function.

        Returns:
            Dict[str, Callable]: Dictionary with function names as keys and function objects as values
        """
        kernel_functions = {}

        # Get all class methods
        for name, method in inspect.getmembers(
            GenericTools, predicate=inspect.isfunction
        ):
            # Skip this method itself and any private/special methods
            if name.startswith("_") or name == "get_all_kernel_functions":
                continue

            # Check if the method has the kernel_function annotation
            # by looking at its __annotations__ attribute
            method_attrs = getattr(method, "__annotations__", {})
            if hasattr(method, "__kernel_function__") or "kernel_function" in str(
                method_attrs
            ):
                kernel_functions[name] = method

        logging.info(
            f"GenericTools.get_all_kernel_functions found {len(kernel_functions)} functions: {list(kernel_functions.keys())}"
        )

        return kernel_functions
