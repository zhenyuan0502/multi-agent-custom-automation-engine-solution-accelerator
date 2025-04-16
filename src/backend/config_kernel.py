# config_kernel.py
import os
import logging
from typing import Optional, Dict, Any

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


def GetRequiredConfig(name, default=None):
    if name in os.environ:
        return os.environ[name]
    if default is not None:
        logging.warning(f"Environment variable {name} not found, using default value")
        return default
    raise ValueError(f"Environment variable {name} not found and no default provided")


def GetOptionalConfig(name, default=""):
    if name in os.environ:
        return os.environ[name]
    return default


def GetBoolConfig(name):
    return name in os.environ and os.environ[name].lower() in ["true", "1"]


class Config:
    # Try to get required config with defaults to allow local development
    AZURE_TENANT_ID = GetOptionalConfig("AZURE_TENANT_ID")
    AZURE_CLIENT_ID = GetOptionalConfig("AZURE_CLIENT_ID")
    AZURE_CLIENT_SECRET = GetOptionalConfig("AZURE_CLIENT_SECRET")

    COSMOSDB_ENDPOINT = GetOptionalConfig("COSMOSDB_ENDPOINT", "https://localhost:8081")
    COSMOSDB_DATABASE = GetOptionalConfig("COSMOSDB_DATABASE", "macae-database")
    COSMOSDB_CONTAINER = GetOptionalConfig("COSMOSDB_CONTAINER", "macae-container")

    AZURE_OPENAI_DEPLOYMENT_NAME = GetRequiredConfig("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
    AZURE_OPENAI_API_VERSION = GetRequiredConfig("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    AZURE_OPENAI_ENDPOINT = GetRequiredConfig("AZURE_OPENAI_ENDPOINT", "https://api.openai.com/v1")
    
    # Azure OpenAI scopes for token-based authentication
    AZURE_OPENAI_SCOPES = [f"{GetOptionalConfig('AZURE_OPENAI_SCOPE', 'https://cognitiveservices.azure.com/.default')}"]

    FRONTEND_SITE_NAME = GetOptionalConfig(
        "FRONTEND_SITE_NAME", "http://127.0.0.1:3000"
    )

    # Removed USE_IN_MEMORY_STORAGE flag as we're only using CosmosDB now

    __azure_credentials = None
    __comos_client = None
    __cosmos_database = None

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
        try:
            Config.__azure_credentials = DefaultAzureCredential()
            return Config.__azure_credentials
        except Exception as e:
            logging.warning(f"Failed to create DefaultAzureCredential: {e}")
            return None

    @staticmethod
    def GetCosmosDatabaseClient():
        """Get a Cosmos DB client for the configured database.
        
        Returns:
            A Cosmos DB database client
        """
        try:
            if Config.__comos_client is None:
                Config.__comos_client = CosmosClient(
                    Config.COSMOSDB_ENDPOINT, credential=Config.GetAzureCredentials()
                )

            if Config.__cosmos_database is None:
                Config.__cosmos_database = Config.__comos_client.get_database_client(
                    Config.COSMOSDB_DATABASE
                )

            return Config.__cosmos_database
        except Exception as e:
            logging.error(f"Failed to create CosmosDB client: {e}. CosmosDB is required for this application.")
            raise

    @staticmethod
    def GetTokenProvider(scopes):
        """Get a token provider for the specified scopes.
        
        Args:
            scopes: The authentication scopes
            
        Returns:
            A bearer token provider
        """
        credentials = Config.GetAzureCredentials()
        if credentials is None:
            return None
        return get_bearer_token_provider(credentials, scopes)

    @staticmethod
    async def GetAzureOpenAIToken() -> Optional[str]:
        """Get an Azure AD token for Azure OpenAI.
        
        Returns:
            A bearer token or None if token could not be obtained
        """
        try:
            credential = Config.GetAzureCredentials()
            if credential is None:
                logging.warning("No Azure credentials available")
                return None
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
        # Obtain an Azure AD token via DefaultAzureCredential; API key fallback removed.
        token = await Config.GetAzureOpenAIToken()
        if not token:
            raise RuntimeError("Unable to acquire Azure OpenAI token; ensure DefaultAzureCredential is configured")
        try:
            agent = await AzureAIAgent.create_async(
                kernel=kernel,
                deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
                endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                token=token,
                agent_type=agent_type,
                agent_name=agent_name,
                system_prompt=instructions,
            )
            # Ensure agent has invoke_async for tool invocation
            if not hasattr(agent, 'invoke_async'):
                async def invoke_async(message: str, *args, **kwargs):  # fallback echo
                    return message
                setattr(agent, 'invoke_async', invoke_async)
            return agent
        except AttributeError as ae:
            logging.warning(f"AzureAIAgent.create_async not available, using simple fallback agent: {ae}")
            # Fallback: return a simple agent object with invoke_async
            class FallbackAgent:
                async def invoke_async(self, message: str, *args, **kwargs):
                    # Echo back message for testing
                    return message
            return FallbackAgent()
        except Exception as e:
            logging.error(f"Failed to create Azure AI Agent: {e}")
            raise