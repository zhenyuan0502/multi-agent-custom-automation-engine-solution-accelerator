# config_kernel.py
import os

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

    FRONTEND_SITE_NAME = GetOptionalConfig(
        "FRONTEND_SITE_NAME", "http://127.0.0.1:3000"
    )

    __azure_credentials = DefaultAzureCredential()
    __comos_client = None
    __cosmos_database = None
    __azure_chat_completion_service = None

    def GetAzureCredentials():
        # If we have specified the credentials in the environment, use them (backwards compatibility)
        if all(
            [Config.AZURE_TENANT_ID, Config.AZURE_CLIENT_ID, Config.AZURE_CLIENT_SECRET]
        ):
            return ClientSecretCredential(
                tenant_id=Config.AZURE_TENANT_ID,
                client_id=Config.AZURE_CLIENT_ID,
                client_secret=Config.AZURE_CLIENT_SECRET,
            )

        # Otherwise, use the default Azure credential which includes managed identity
        return Config.__azure_credentials

    # Gives us a cached approach to DB access
    def GetCosmosDatabaseClient():
        # TODO: Today this is a single DB, we might want to support multiple DBs in the future
        if Config.__comos_client is None:
            Config.__comos_client = CosmosClient(
                Config.COSMOSDB_ENDPOINT, Config.GetAzureCredentials()
            )

        if Config.__cosmos_database is None:
            Config.__cosmos_database = Config.__comos_client.get_database_client(
                Config.COSMOSDB_DATABASE
            )

        return Config.__cosmos_database

    def GetTokenProvider(scopes):
        return get_bearer_token_provider(Config.GetAzureCredentials(), scopes)

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

        if Config.AZURE_OPENAI_API_KEY == "":
            # Use Azure AD token-based authentication
            # Note: Semantic Kernel's AzureChatCompletion doesn't directly support token providers
            # This would need to be implemented in a custom connector or using a different approach
            # For now, we'll raise an error in this case
            raise NotImplementedError(
                "Token-based authentication not yet implemented for Semantic Kernel. Please provide an API key."
            )
        else:
            # Use API key authentication
            Config.__azure_chat_completion_service = AzureChatCompletion(
                service_id=service_id,
                deployment_name=deployment_name,
                endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )

        return Config.__azure_chat_completion_service
    
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