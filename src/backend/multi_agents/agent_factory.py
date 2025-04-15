"""Factory for creating agents in the Multi-Agent Custom Automation Engine."""

import logging
from typing import Dict, List, Callable, Any, Optional, Type
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction

from models.agent_types import AgentType
from multi_agents.agent_base import BaseAgent
from multi_agents.agent_config import AgentBaseConfig

# Import all agent implementations
from multi_agents.hr_agent import HrAgent
from multi_agents.human_agent import HumanAgent 
from multi_agents.marketing_agent import MarketingAgent
from multi_agents.generic_agent import GenericAgent
from multi_agents.planner_agent import PlannerAgent
from multi_agents.tech_support_agent import TechSupportAgent
from multi_agents.procurement_agent import ProcurementAgent
from multi_agents.product_agent import ProductAgent
from multi_agents.group_chat_manager import GroupChatManager

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating agents in the Multi-Agent Custom Automation Engine."""

    # Mapping of agent types to their implementation classes
    _agent_classes: Dict[AgentType, Type[BaseAgent]] = {
        AgentType.HR: HrAgent,
        AgentType.MARKETING: MarketingAgent,
        AgentType.PRODUCT: ProductAgent,
        AgentType.PROCUREMENT: ProcurementAgent,
        AgentType.TECH_SUPPORT: TechSupportAgent,
        AgentType.GENERIC: GenericAgent,
        AgentType.HUMAN: HumanAgent,
        AgentType.PLANNER: PlannerAgent,
        AgentType.GROUP_CHAT_MANAGER: GroupChatManager,
    }

    # Mapping of agent types to functions that provide their tools
    _tool_getters: Dict[AgentType, Callable[[Kernel], List[KernelFunction]]] = {}

    # Cache of agent instances by session_id and agent_type
    _agent_cache: Dict[str, Dict[AgentType, BaseAgent]] = {}

    @classmethod
    def register_agent_class(
        cls, agent_type: AgentType, agent_class: Type[BaseAgent]
    ) -> None:
        """Register a new agent class with the factory.
        
        Args:
            agent_type: The type of agent to register
            agent_class: The class to use for this agent type
        """
        cls._agent_classes[agent_type] = agent_class
        logger.info(
            f"Registered agent class {agent_class.__name__} for type {agent_type.value}"
        )

    @classmethod
    def register_tool_getter(
        cls, agent_type: AgentType, tool_getter: Callable[[Kernel], List[KernelFunction]]
    ) -> None:
        """Register a tool getter function for an agent type.
        
        Args:
            agent_type: The type of agent
            tool_getter: A function that returns a list of tools for the agent
        """
        cls._tool_getters[agent_type] = tool_getter
        logger.info(f"Registered tool getter for agent type {agent_type.value}")

    @classmethod
    async def create_agent(
        cls,
        agent_type: AgentType,
        session_id: str,
        user_id: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        **kwargs
    ) -> BaseAgent:
        """Create an agent of the specified type.
        
        Args:
            agent_type: The type of agent to create
            session_id: The session ID
            user_id: The user ID
            temperature: The temperature to use for the agent
            system_message: Optional system message for the agent
            **kwargs: Additional parameters to pass to the agent constructor
            
        Returns:
            An instance of the specified agent type
            
        Raises:
            ValueError: If the agent type is unknown
        """
        # Check if we already have an agent in the cache
        if session_id in cls._agent_cache and agent_type in cls._agent_cache[session_id]:
            return cls._agent_cache[session_id][agent_type]
            
        # Get the agent class
        agent_class = cls._agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        # Create a kernel and memory store
        kernel = AgentBaseConfig.create_kernel()
        memory_store = await AgentBaseConfig.create_memory_store(session_id, user_id)
        
        # Create agent configuration
        config = AgentBaseConfig(
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store
        )
        
        # Get tools for this agent type
        tools = []
        if agent_type in cls._tool_getters:
            tools = cls._tool_getters[agent_type](kernel)
            
        # Create the agent instance
        try:
            agent = agent_class(
                config=config,
                tools=tools,
                temperature=temperature,
                system_message=system_message,
                **kwargs
            )
        except Exception as e:
            logger.error(
                f"Error creating agent of type {agent_type} with parameters: {e}"
            )
            raise
            
        # Cache the agent instance
        if session_id not in cls._agent_cache:
            cls._agent_cache[session_id] = {}
        cls._agent_cache[session_id][agent_type] = agent
        
        return agent
        
    @classmethod
    async def create_all_agents(
        cls,
        session_id: str,
        user_id: str,
        temperature: float = 0.7
    ) -> Dict[AgentType, BaseAgent]:
        """Create all agent types for a session.
        
        Args:
            session_id: The session ID
            user_id: The user ID
            temperature: The temperature to use for the agents
            
        Returns:
            Dictionary mapping agent types to agent instances
        """
        # Check if we already have all agents in the cache
        if session_id in cls._agent_cache and len(cls._agent_cache[session_id]) == len(cls._agent_classes):
            return cls._agent_cache[session_id]
            
        # Create each agent type
        agents = {}
        for agent_type in cls._agent_classes.keys():
            agents[agent_type] = await cls.create_agent(
                agent_type=agent_type,
                session_id=session_id,
                user_id=user_id,
                temperature=temperature
            )
            
        return agents
        
    @classmethod
    def get_agent_class(cls, agent_type: AgentType) -> Type[BaseAgent]:
        """Get the agent class for the specified type.
        
        Args:
            agent_type: The agent type
            
        Returns:
            The agent class
            
        Raises:
            ValueError: If the agent type is unknown
        """
        agent_class = cls._agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class
        
    @classmethod
    def clear_cache(cls, session_id: Optional[str] = None) -> None:
        """Clear the agent cache.
        
        Args:
            session_id: If provided, clear only this session's cache
        """
        if session_id:
            if session_id in cls._agent_cache:
                del cls._agent_cache[session_id]
                logger.info(f"Cleared agent cache for session {session_id}")
        else:
            cls._agent_cache.clear()
            logger.info("Cleared all agent caches")