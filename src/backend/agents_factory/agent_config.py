"""Configuration class for the agents in the Multi-Agent Custom Automation Engine.

This class loads configuration values from environment variables and provides
properties to access them. It also stores the semantic kernel instance, memory store,
and other configuration needed by agents.
"""

import logging
import os
from typing import Dict, Any, Optional

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.memory.azure_cosmos_db import AzureCosmosDBMemoryStore

from config_kernel import Config
from context.cosmos_memory_kernel import CosmosMemoryContext
from context.cosmos_memory import CosmosMemory


class AgentBaseConfig:
    """Base configuration for agents."""

    # Model deployment names
    MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME", "gpt-35-turbo")
    
    # API configuration
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    # Cosmos DB configuration
    COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
    COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")
    COSMOS_DB = os.getenv("AZURE_COSMOS_DB", "MACAE")
    COSMOS_CONTAINER = os.getenv("AZURE_COSMOS_CONTAINER", "memory")

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext
    ):
        """Initialize the agent configuration.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The session ID
            user_id: The user ID
            memory_store: The memory store
        """
        self.kernel = kernel
        self.session_id = session_id
        self.user_id = user_id
        self.memory_store = memory_store

    @classmethod
    def create_kernel(cls) -> sk.Kernel:
        """Create a semantic kernel instance.
        
        Returns:
            A configured semantic kernel instance
        """
        kernel = sk.Kernel()
        
        # Set up OpenAI service for the kernel
        if cls.OPENAI_ENDPOINT and cls.OPENAI_API_KEY:
            kernel.add_service(
                AzureChatCompletion(
                    service_id="azure_chat_completion",
                    endpoint=cls.OPENAI_ENDPOINT,
                    api_key=cls.OPENAI_API_KEY,
                    api_version=cls.OPENAI_API_VERSION,
                    deployment_name=cls.MODEL_DEPLOYMENT_NAME,
                    log=logging.getLogger("semantic_kernel.kernel"),
                )
            )
        else:
            logging.warning("Azure OpenAI configuration missing. Kernel will have limited functionality.")
        
        return kernel

    @classmethod
    async def create_memory_store(cls, session_id: str, user_id: str) -> CosmosMemoryContext:
        """Create a memory store for the agent.
        
        Args:
            session_id: The session ID
            user_id: The user ID
            
        Returns:
            A configured memory store
        """
        # Create Cosmos DB memory store if configuration is available
        if cls.COSMOS_ENDPOINT and cls.COSMOS_KEY:
            cosmos_memory = CosmosMemory(
                cosmos_endpoint=cls.COSMOS_ENDPOINT,
                cosmos_key=cls.COSMOS_KEY,
                database_name=cls.COSMOS_DB,
                container_name=cls.COSMOS_CONTAINER
            )
            
            memory_store = CosmosMemoryContext(
                cosmos_memory=cosmos_memory,
                session_id=session_id,
                user_id=user_id
            )
            
            return memory_store
        else:
            logging.warning("Cosmos DB configuration missing. Using in-memory store instead.")
            # Create an in-memory store as fallback
            # This is useful for local development without Cosmos DB
            from context.cosmos_memory_kernel import InMemoryContext
            return InMemoryContext(session_id, user_id)

    def get_model_config(self) -> Dict[str, Any]:
        """Get the model configuration.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            "deployment_name": self.MODEL_DEPLOYMENT_NAME,
            "endpoint": self.OPENAI_ENDPOINT,
            "api_key": self.OPENAI_API_KEY,
            "api_version": self.OPENAI_API_VERSION
        }

    def clone_with_session(self, session_id: str) -> 'AgentBaseConfig':
        """Create a new configuration with a different session ID.
        
        Args:
            session_id: The new session ID
            
        Returns:
            A new configuration instance
        """
        return AgentBaseConfig(
            kernel=self.kernel,
            session_id=session_id,
            user_id=self.user_id,
            memory_store=self.memory_store
        )