# config_kernel.py
import os
import logging
from typing import Optional

# Import Semantic Kernel instead of AutoGen
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import (
    ClientSecretCredential,
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

    @staticmethod
    def GetAzureCredentials():
        # Cache the credentials object
        if Config.__azure_credentials is not None:
            return Config.__azure_credentials
            
        # If we have specified the credentials in the environment, use them
        if all(
            [Config.AZURE_TENANT_ID, Config.AZURE_CLIENT_ID, Config.AZURE_CLIENT_SECRET]
        ):
            Config.__azure_credentials = ClientSecretCredential(
                tenant_id=Config.AZURE_TENANT_ID,
                client_id=Config.AZURE_CLIENT_ID,
                client_secret=Config.AZURE_CLIENT_SECRET,
            )
        else:
            # Use the default Azure credential which includes managed identity
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
        api_key = Config.AZURE_OPENAI_API_KEY
        api_version = Config.AZURE_OPENAI_API_VERSION

        # Try to use token-based auth if API key is not provided
        use_token_auth = not api_key
        
        if use_token_auth:
            try:
                # Create a custom AzureChatCompletion that supports tokens
                # Note: Semantic Kernel's current implementation may not fully support
                # token-based authentication directly, so this is a placeholder for future updates
                logging.warning("Using token-based authentication for Azure OpenAI")
                Config.__azure_chat_completion_service = AzureChatCompletionWithToken(
                    service_id=service_id,
                    deployment_name=deployment_name,
                    endpoint=endpoint, 
                    api_version=api_version
                )
            except Exception as e:
                logging.error(f"Failed to initialize token-based Azure OpenAI: {e}")
                logging.warning("Falling back to API key authentication")
                use_token_auth = False
                
        # If token auth failed or wasn't attempted, use API key
        if not use_token_auth:
            if not api_key:
                raise ValueError("No API key provided for Azure OpenAI and token authentication failed")
                
            Config.__azure_chat_completion_service = AzureChatCompletion(
                service_id=service_id,
                deployment_name=deployment_name,
                endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )

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


# This is a placeholder for a future implementation that supports token-based authentication
# The actual implementation would depend on Semantic Kernel's support for token auth
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
        
        # Note: In a real implementation, you would override the methods that make
        # HTTP requests to Azure OpenAI to include the Authorization header with the token
        # For now, this is just a placeholder until Semantic Kernel provides better support
        logging.warning("Token-based authentication for Azure OpenAI is not fully implemented")