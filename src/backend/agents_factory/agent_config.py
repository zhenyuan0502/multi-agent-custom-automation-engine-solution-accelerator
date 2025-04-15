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

from azure.identity.aio import ClientSecretCredential, DefaultAzureCredential

from config_kernel import Config
from context.cosmos_memory_kernel import CosmosMemoryContext


class AgentBaseConfig:
    """Base configuration for agents."""
    
    # Use Config class to get values instead of direct environment variables
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
        # Use Config class to create kernel with properly configured services
        try:
            kernel = Config.CreateKernel()
            return kernel
        except Exception as e:
            logging.error(f"Error creating kernel: {e}")
            # Provide a fallback kernel with limited functionality
            logging.warning("Creating kernel with limited functionality")
            kernel = sk.Kernel()
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
        # Use Config class to get credentials and connection info
        try:
            # Import here to avoid circular import issues
            from context.cosmos_memory import CosmosMemory
            
            # Get Cosmos DB credentials and endpoints from Config
            cosmos_memory = CosmosMemory(
                cosmos_endpoint=Config.COSMOSDB_ENDPOINT,
                database_name=Config.COSMOSDB_DATABASE,
                container_name=Config.COSMOSDB_CONTAINER,
                credential=Config.GetAzureCredentials()
            )
            
            memory_store = CosmosMemoryContext(
                cosmos_memory=cosmos_memory,
                session_id=session_id,
                user_id=user_id
            )
            
            return memory_store
        except Exception as e:
            logging.error(f"Error creating memory store: {e}")
            logging.warning("Using in-memory store instead")
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
            "deployment_name": Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            "endpoint": Config.AZURE_OPENAI_ENDPOINT,
            "api_version": Config.AZURE_OPENAI_API_VERSION,
            # Note: No API key here - using Azure credentials instead
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