# config_kernel.py
import os
import logging
from typing import Optional

# Import Semantic Kernel and Azure AI Agent
from semantic_kernel import Kernel
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from dotenv import load_dotenv

load_dotenv()


def GetRequiredConfig(name):
    return os.environ[name]


def GetOptionalConfig(name, default=""):
    if name in os.environ:
        return os.environ[name]
    return default


def GetBoolConfig(name):
    return name in os.environ and os.environ[name].lower() in ["true", "1"]


class Config:
    AZURE_TENANT_ID = GetOptionalConfig("AZURE_TENANT_ID")
    AZURE_CLIENT_ID = GetOptionalConfig("AZURE_CLIENT_ID")
    AZURE_CLIENT_SECRET = GetOptionalConfig("AZURE_CLIENT_SECRET")

    COSMOSDB_ENDPOINT = GetRequiredConfig("COSMOSDB_ENDPOINT")
    COSMOSDB_DATABASE = GetRequiredConfig("COSMOSDB_DATABASE")
    COSMOSDB_CONTAINER = GetRequiredConfig("COSMOSDB_CONTAINER")

    AZURE_OPENAI_DEPLOYMENT_NAME = GetRequiredConfig("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = GetRequiredConfig("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_ENDPOINT = GetRequiredConfig("AZURE_OPENAI_ENDPOINT")
    
    # Azure OpenAI scopes for token-based authentication
    AZURE_OPENAI_SCOPES = [f"{GetOptionalConfig('AZURE_OPENAI_SCOPE', 'https://cognitiveservices.azure.com/.default')}"]

    FRONTEND_SITE_NAME = GetOptionalConfig(
        "FRONTEND_SITE_NAME", "http://127.0.0.1:3000"
    )

    __azure_credentials = None
    __comos_client = None
    __cosmos_database = None
    __azure_ai_agent_config = None

    @staticmethod
    def GetAzureCredentials():
        """Get Azure credentials using DefaultAzureCredential.
        
        Returns:
            DefaultAzureCredential instance for Azure authentication
        """
        # Cache the credentials object
        if Config.__azure_credentials is not None:
            return Config.__azure_credentials

        # Always use DefaultAzureCredential
        Config.__azure_credentials = DefaultAzureCredential()
        return Config.__azure_credentials

    @staticmethod
    def GetCosmosDatabaseClient():
        """Get a Cosmos DB client for the configured database.
        
        Returns:
            A Cosmos DB database client
        """
        if Config.__comos_client is None:
            Config.__comos_client = CosmosClient(
                Config.COSMOSDB_ENDPOINT, credential=Config.GetAzureCredentials()
            )

        if Config.__cosmos_database is None:
            Config.__cosmos_database = Config.__comos_client.get_database_client(
                Config.COSMOSDB_DATABASE
            )

        return Config.__cosmos_database

    @staticmethod
    def GetTokenProvider(scopes):
        """Get a token provider for the specified scopes.
        
        Args:
            scopes: The authentication scopes
            
        Returns:
            A bearer token provider
        """
        return get_bearer_token_provider(Config.GetAzureCredentials(), scopes)

    @staticmethod
    async def GetAzureOpenAIToken() -> Optional[str]:
        """Get an Azure AD token for Azure OpenAI.
        
        Returns:
            A bearer token or None if token could not be obtained
        """
        try:
            credential = Config.GetAzureCredentials()
            token = await credential.get_token(*Config.AZURE_OPENAI_SCOPES)
            return token.token
        except Exception as e:
            logging.error(f"Failed to get Azure OpenAI token: {e}")
            return None
    
    @staticmethod
    def CreateKernel():
        """
        Creates a new Semantic Kernel instance.
        
        Returns:
            A new Semantic Kernel instance
        """
        kernel = Kernel()
        return kernel
    
    @staticmethod
    async def CreateAzureAIAgent(kernel: Kernel, agent_name: str, instructions: str, agent_type: str = "assistant"):
        """
        Creates a new Azure AI Agent with the specified name and instructions.
        
        Args:
            kernel: The Semantic Kernel instance
            agent_name: The name of the agent
            instructions: The system message / instructions for the agent
            agent_type: The type of agent (defaults to "assistant")
            
        Returns:
            A new AzureAIAgent instance
        """
        # Get token for authentication
        token = await Config.GetAzureOpenAIToken()
        if not token:
            raise ValueError("Failed to obtain Azure OpenAI authentication token")
            
        # Create the Azure AI Agent
        agent = await AzureAIAgent.create_async(
            kernel=kernel,
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            token=token,  # Use token for authentication
            agent_type=agent_type,
            agent_name=agent_name,
            system_prompt=instructions,
        )
        
        return agent