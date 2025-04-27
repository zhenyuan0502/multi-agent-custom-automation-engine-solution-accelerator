import logging
import json
from datetime import datetime
import re
from typing import Dict, List, Optional, Any, Tuple

import semantic_kernel as sk
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.agents import AgentGroupChat  # pylint: disable=E0611

from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    TerminationStrategy,
)

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    ActionRequest,
    ActionResponse,
    AgentMessage,
    Step,
    StepStatus,
    PlanStatus,
    HumanFeedbackStatus,
    InputTask,
    Plan,
)
from models.messages_kernel import AgentType
from event_utils import track_event_if_configured

class GroupChatManager(BaseAgent):
    """GroupChatManager agent implementation using Semantic Kernel.

    This agent creates and manages plans based on user tasks, breaking them down into steps
    that can be executed by specialized agents to achieve the user's goal.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.GROUP_CHAT_MANAGER.value,
        available_agents: List[str] = None,
        agent_tools_list: List[str] = None,
        agent_instances: Optional[Dict[str, BaseAgent]] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the GroupChatManager Agent.

        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: Optional list of tools for this agent
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "GroupChatManagerAgent")
            config_path: Optional path to the configuration file
            available_agents: List of available agent names for creating steps
            agent_tools_list: List of available tools across all agents
            agent_instances: Dictionary of agent instances available to the GroupChatManager
            client: Optional client instance (passed to BaseAgent)
            definition: Optional definition instance (passed to BaseAgent)
        """
        # Default system message if not provided
        if not system_message:
            system_message = "You are a GroupChatManager agent responsible for creating and managing plans. You analyze tasks, break them down into steps, and assign them to the appropriate specialized agents."

        # Initialize the base agent
        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            agent_type=AgentType.GROUP_CHAT_MANAGER.value,  # Use GroupChatManager_tools.json if available
            client=client,
            definition=definition,
        )

        # Store additional GroupChatManager-specific attributes
        self._available_agents = available_agents or [
            AgentType.HUMAN.value,
            AgentType.HR.value,
            AgentType.MARKETING.value,
            AgentType.PRODUCT.value,
            AgentType.PROCUREMENT.value,
            AgentType.TECH_SUPPORT.value,
            AgentType.GENERIC.value,
        ]
        self._agent_tools_list = agent_tools_list or []
        self._agent_instances = agent_instances or {}

        # Create the Azure AI Agent for group chat operations
        # This will be initialized in async_init
        self._azure_ai_agent = None