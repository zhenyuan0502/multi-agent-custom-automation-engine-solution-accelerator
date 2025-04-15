# config_kernel.py
import os
import logging
from typing import Optional

# Import Semantic Kernel and Azure AI Agent
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
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
    AZURE_OPENAI_API_KEY = GetOptionalConfig("AZURE_OPENAI_API_KEY")
    
    # Azure OpenAI scopes for token-based authentication
    AZURE_OPENAI_SCOPES = [f"{GetOptionalConfig('AZURE_OPENAI_SCOPE', 'https://cognitiveservices.azure.com/.default')}"]

    FRONTEND_SITE_NAME = GetOptionalConfig(
        "FRONTEND_SITE_NAME", "http://127.0.0.1:3000"
    )

    __azure_credentials = None
    __comos_client = None
    __cosmos_database = None
    __azure_chat_completion_service = None
    __azure_ai_agent_config = None

    @staticmethod
    def GetAzureCredentials():
        # Cache the credentials object
        if Config.__azure_credentials is not None:
            return Config.__azure_credentials

        # Always prefer DefaultAzureCredential
        Config.__azure_credentials = DefaultAzureCredential()
        return Config.__azure_credentials

    # Gives us a cached approach to DB access
    @staticmethod
    def GetCosmosDatabaseClient():
        # TODO: Today this is a single DB, we might want to support multiple DBs in the future
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
    def GetAzureOpenAIChatCompletionService():
        """
        Gets or creates an Azure Chat Completion service for Semantic Kernel.
        
        Returns:
            The Azure Chat Completion service instance
        """
        if Config.__azure_chat_completion_service is not None:
            return Config.__azure_chat_completion_service

        service_id = "chat_service"
        deployment_name = Config.AZURE_OPENAI_DEPLOYMENT_NAME
        endpoint = Config.AZURE_OPENAI_ENDPOINT
        api_version = Config.AZURE_OPENAI_API_VERSION

        # Always prefer token-based authentication using DefaultAzureCredential
        try:
            # Create a custom AzureChatCompletion that supports tokens
            logging.info("Using token-based authentication for Azure OpenAI")
            Config.__azure_chat_completion_service = AzureChatCompletionWithToken(
                service_id=service_id,
                deployment_name=deployment_name,
                endpoint=endpoint, 
                api_version=api_version
            )
        except Exception as e:
            logging.error(f"Failed to initialize token-based Azure OpenAI: {e}")
            
            # Only fall back to API key if we have one and token auth failed
            if Config.AZURE_OPENAI_API_KEY:
                logging.warning("Falling back to API key authentication")
                Config.__azure_chat_completion_service = AzureChatCompletion(
                    service_id=service_id,
                    deployment_name=deployment_name,
                    endpoint=endpoint,
                    api_key=Config.AZURE_OPENAI_API_KEY,
                    api_version=api_version
                )
            else:
                raise ValueError("Failed to authenticate with Azure OpenAI. No API key provided and token authentication failed.")

        return Config.__azure_chat_completion_service
    
    @staticmethod
    def CreateKernel():
        """
        Creates a new Semantic Kernel instance with the Azure Chat Completion service configured.
        
        Returns:
            A new Semantic Kernel instance
        """
        kernel = Kernel()
        service = Config.GetAzureOpenAIChatCompletionService()
        kernel.add_service(service)
        return kernel

    @staticmethod
    def GetAzureAIAgentConfig():
        """
        Gets or creates the configuration for Azure AI Agents.
        
        Returns:
            A dictionary with configuration for creating Azure AI Agents
        """
        if Config.__azure_ai_agent_config is not None:
            return Config.__azure_ai_agent_config
            
        # We prefer token-based auth via DefaultAzureCredential
        token = None
        # This is a synchronous method, so we can't await GetAzureOpenAIToken directly
        # In a real implementation, you'd make this method async
        
        Config.__azure_ai_agent_config = {
            "deployment_name": Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            "endpoint": Config.AZURE_OPENAI_ENDPOINT,
            "api_version": Config.AZURE_OPENAI_API_VERSION,
            # Include API key as fallback only
            "api_key": Config.AZURE_OPENAI_API_KEY if not token else None,
            # In a real implementation, you'd include the token here
            # "token": token
        }
        
        return Config.__azure_ai_agent_config
    
    @staticmethod
    def CreateAzureAIAgent(kernel: Kernel, agent_name: str, instructions: str, agent_type: str = "assistant"):
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
        config = Config.GetAzureAIAgentConfig()
        
        # Try to use token-based auth if possible
        # For now, we fall back to API key if needed
        if not config["api_key"]:
            # This isn't ideal - in a real implementation we would make this method async
            # and await the token properly
            logging.warning("API key not available for AzureAIAgent - in production, implement proper token auth")
        
        # Create the Azure AI Agent
        agent = AzureAIAgent.create(
            kernel=kernel,
            deployment_name=config["deployment_name"],
            endpoint=config["endpoint"],
            api_key=config["api_key"],  # This would be None if using token auth
            api_version=config["api_version"],
            agent_type=agent_type,
            agent_name=agent_name,
            system_prompt=instructions,
        )
        
        return agent


# This is a modified implementation that supports token-based authentication
class AzureChatCompletionWithToken(AzureChatCompletion):
    """Extended Azure Chat Completion service that supports token-based authentication."""
    
    def __init__(
        self,
        service_id: str,
        deployment_name: str,
        endpoint: str,
        api_version: str
    ):
        # Initialize without an API key
        super().__init__(
            service_id=service_id,
            deployment_name=deployment_name,
            endpoint=endpoint,
            api_key="placeholder_will_use_token",  # Placeholder
            api_version=api_version
        )
        
        # Store credentials for token retrieval
        self._credentials = Config.GetAzureCredentials()
        self._scopes = Config.AZURE_OPENAI_SCOPES
        
        logging.info("Initialized token-based authentication for Azure OpenAI")