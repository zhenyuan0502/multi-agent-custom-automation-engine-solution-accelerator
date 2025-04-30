import inspect
from typing import Annotated, Callable, Dict

from semantic_kernel.functions import kernel_function

formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown. Then write AGENT SUMMARY: and then include a summary of what you did."


class HumanTools:
    # Define Human tools (functions)
    @staticmethod
    @kernel_function(
        description="Parse and process HumanFeedback JSON to update the step status and record feedback."
    )
    async def handle_human_feedback(human_feedback_json: str) -> str:
        return (
            f"##### Human Feedback Processing\n"
            f"**Feedback JSON:** {human_feedback_json}\n\n"
            f"Human feedback processed successfully\n"
            f"{formatting_instructions}"
        )

    @staticmethod
    @kernel_function(
        description="Provide clarification on a plan, storing the user's response."
    )
    async def provide_clarification(session_id: str, clarification_text: str) -> str:
        return (
            f"##### Clarification Provided\n"
            f"**Session ID:** {session_id}\n"
            f"**Clarification:** {clarification_text}\n\n"
            f"Clarification provided for plan in session {session_id}\n"
            f"{formatting_instructions}"
        )

    @staticmethod
    def get_all_kernel_functions() -> Dict[str, Callable]:
        """
        Returns a dictionary of all methods in this class that have the @kernel_function annotation.
        This function itself is not annotated with @kernel_function.

        Returns:
            Dict[str, Callable]: Dictionary with function names as keys and function objects as values
        """
        kernel_functions = {}

        # Get all class methods
        for name, method in inspect.getmembers(
            HumanTools, predicate=inspect.isfunction
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

        return kernel_functions
