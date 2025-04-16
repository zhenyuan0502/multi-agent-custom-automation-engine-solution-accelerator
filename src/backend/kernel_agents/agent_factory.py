"""Factory for creating agents in the Multi-Agent Custom Automation Engine."""

import logging
from typing import Dict, List, Callable, Any, Optional, Type
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent

from models.agent_types import AgentType
from kernel_agents.agent_base import BaseAgent
from config_kernel import Config

# Import all specialized agent implementations
from kernel_agents.hr_agent import HrAgent
from kernel_agents.human_agent import HumanAgent 
from kernel_agents.marketing_agent import MarketingAgent
from kernel_agents.generic_agent import GenericAgent
from kernel_agents.planner_agent import PlannerAgent
from kernel_agents.tech_support_agent import TechSupportAgent
from kernel_agents.procurement_agent import ProcurementAgent
from kernel_agents.product_agent import ProductAgent
from kernel_agents.group_chat_manager import GroupChatManager

from context.cosmos_memory_kernel import CosmosMemoryContext

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

    # Mapping of agent types to their string identifiers (for automatic tool loading)
    _agent_type_strings: Dict[AgentType, str] = {
        AgentType.HR: "hr",
        AgentType.MARKETING: "marketing",
        AgentType.PRODUCT: "product",
        AgentType.PROCUREMENT: "procurement",
        AgentType.TECH_SUPPORT: "tech_support",
        AgentType.GENERIC: "generic",
        AgentType.HUMAN: "human",
        AgentType.PLANNER: "planner",
        AgentType.GROUP_CHAT_MANAGER: "group_chat_manager",
    }

    # System messages for each agent type
    _agent_system_messages: Dict[AgentType, str] = {
        AgentType.HR: "You are an HR assistant helping with human resource related tasks.",
        AgentType.MARKETING: "You are a marketing expert helping with marketing related tasks.",
        AgentType.PRODUCT: "You are a product expert helping with product related tasks.",
        AgentType.PROCUREMENT: "You are a procurement expert helping with procurement related tasks.",
        AgentType.TECH_SUPPORT: "You are a technical support expert helping with technical issues.",
        AgentType.GENERIC: "You are a helpful assistant ready to help with various tasks.",
        AgentType.HUMAN: "You are representing a human user in the conversation.",
        AgentType.PLANNER: "You are a planner agent responsible for creating and managing plans.",
        AgentType.GROUP_CHAT_MANAGER: "You are a group chat manager coordinating the conversation between different agents.",
    }

    # Cache of agent instances by session_id and agent_type
    _agent_cache: Dict[str, Dict[AgentType, BaseAgent]] = {}
    
    # Cache of Azure AI Agent instances
    _azure_ai_agent_cache: Dict[str, Dict[str, AzureAIAgent]] = {}

    @classmethod
    def register_agent_class(
        cls, agent_type: AgentType, agent_class: Type[BaseAgent], agent_type_string: Optional[str] = None,
        system_message: Optional[str] = None
    ) -> None:
        """Register a new agent class with the factory.
        
        Args:
            agent_type: The type of agent to register
            agent_class: The class to use for this agent type
            agent_type_string: Optional string identifier for the agent type (for tool loading)
            system_message: Optional system message for the agent
        """
        cls._agent_classes[agent_type] = agent_class
        if agent_type_string:
            cls._agent_type_strings[agent_type] = agent_type_string
        if system_message:
            cls._agent_system_messages[agent_type] = system_message
        logger.info(
            f"Registered agent class {agent_class.__name__} for type {agent_type.value}"
        )

    @classmethod
    async def create_agent(
        cls,
        agent_type: AgentType,
        session_id: str,
        user_id: str,
        temperature: float = 0.0,
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
            
        # Create memory store
        memory_store = CosmosMemoryContext(session_id, user_id)
        
        # Create a kernel
        kernel = Config.CreateKernel()
        
        # Use default system message if none provided
        if system_message is None:
            system_message = cls._agent_system_messages.get(
                agent_type, 
                f"You are a helpful AI assistant specialized in {cls._agent_type_strings.get(agent_type, 'general')} tasks."
            )
        
        # Special handling for GenericAgent - directly use its get_generic_tools function
        if agent_type == AgentType.GENERIC:
            from kernel_agents.generic_agent import get_generic_tools
            tools = get_generic_tools(kernel)
        else:
            # For other agent types, use the standard tool loading mechanism
            agent_type_str = cls._agent_type_strings.get(agent_type, agent_type.value.lower())
            tools = await cls._load_tools_for_agent(kernel, agent_type_str)
        
        # Create the agent instance
        try:
            agent = agent_class(
                agent_name=cls._agent_type_strings.get(agent_type, agent_type.value.lower()),
                kernel=kernel,
                session_id=session_id,
                user_id=user_id,
                memory_store=memory_store,
                tools=tools,
                system_message=system_message,
                **kwargs
            )
            
            # Initialize the agent asynchronously
            await agent.async_init()
            
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
    async def create_azure_ai_agent(
        cls,
        agent_name: str,
        session_id: str,
        system_prompt: str,
        tools: List[KernelFunction] = None
    ) -> AzureAIAgent:
        """Create an Azure AI Agent.
        
        Args:
            agent_name: The name of the agent
            session_id: The session ID
            system_prompt: The system prompt for the agent
            tools: Optional list of tools for the agent
            
        Returns:
            An Azure AI Agent instance
        """
        # Check if we already have an agent in the cache
        cache_key = f"{session_id}_{agent_name}"
        if session_id in cls._azure_ai_agent_cache and cache_key in cls._azure_ai_agent_cache[session_id]:
            # If tools are provided, make sure they are registered with the cached agent
            agent = cls._azure_ai_agent_cache[session_id][cache_key]
            if tools:
                for tool in tools:
                    agent.add_function(tool)
            return agent
        
        # Create a kernel
        kernel = Config.CreateKernel()
        
        # Create the Azure AI Agent
        agent = Config.CreateAzureAIAgent(
            kernel=kernel,
            agent_name=agent_name,
            instructions=system_prompt
        )
        
        # Register tools if provided
        if tools:
            for tool in tools:
                agent.add_function(tool)
                
        # Cache the agent instance
        if session_id not in cls._azure_ai_agent_cache:
            cls._azure_ai_agent_cache[session_id] = {}
        cls._azure_ai_agent_cache[session_id][cache_key] = agent
        
        return agent
        
    @classmethod
    async def _load_tools_for_agent(cls, kernel: Kernel, agent_type: str) -> List[KernelFunction]:
        """Load tools for an agent from the tools directory.
        
        This is a placeholder implementation. In a real system, you would load
        tool configurations from JSON files and register them with the kernel.
        
        Args:
            kernel: The semantic kernel instance
            agent_type: The agent type string identifier
            
        Returns:
            A list of kernel functions for the agent
        """
        try:
            # Try to use the BaseAgent's tool loading mechanism
            return BaseAgent.get_tools_from_config(kernel, agent_type)
        except Exception as e:
            # If it fails, create and return a dummy function to avoid initialization failure
            logger.warning(f"Failed to load tools for {agent_type}, using fallback: {e}")
            
            # Create a dummy function that always works
            from semantic_kernel.functions.kernel_function import KernelFunction
            
            function_name = "dummy_function"
            dummy_prompt = f"You are helping with {agent_type} tasks.\n\n{{{{$input}}}}"
            
            dummy_function = KernelFunction.from_prompt(
                function_name=function_name,
                plugin_name=agent_type,
                description=f"Fallback function for {agent_type}",
                prompt=dummy_prompt  # This is the key - providing a prompt parameter
            )
            
            # Add the function to the kernel
            kernel.add_function(dummy_function)
            
            return [dummy_function]
        
    @classmethod
    async def create_all_agents(
        cls,
        session_id: str,
        user_id: str,
        temperature: float = 0.0
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
            if session_id in cls._azure_ai_agent_cache:
                del cls._azure_ai_agent_cache[session_id]
                logger.info(f"Cleared Azure AI agent cache for session {session_id}")
        else:
            cls._agent_cache.clear()
            cls._azure_ai_agent_cache.clear()
            logger.info("Cleared all agent caches")