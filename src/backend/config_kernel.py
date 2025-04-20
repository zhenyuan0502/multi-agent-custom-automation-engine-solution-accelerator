# config_kernel.py
import os
import logging
from typing import Optional

# Import Semantic Kernel and Azure AI Agent
from semantic_kernel import Kernel
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential
from azure.ai.projects.aio import AIProjectClient
from dotenv import load_dotenv

load_dotenv()


def GetRequiredConfig(name, default=None):
    if name in os.environ:
        return os.environ[name]
    if default is not None:
        logging.warning("Environment variable %s not found, using default value", name)
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

    AZURE_AI_SUBSCRIPTION_ID = GetRequiredConfig("AZURE_AI_SUBSCRIPTION_ID")
    AZURE_AI_RESOURCE_GROUP = GetRequiredConfig("AZURE_AI_RESOURCE_GROUP")
    AZURE_AI_PROJECT_NAME = GetRequiredConfig("AZURE_AI_PROJECT_NAME")
    AZURE_AI_AGENT_PROJECT_CONNECTION_STRING = GetRequiredConfig("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING")

    __azure_credentials = None
    __comos_client = None
    __cosmos_database = None
    __ai_project_client = None

    @staticmethod
    def GetAzureCredentials():
        """Get Azure credentials using DefaultAzureCredential.
        
        Returns:
            DefaultAzureCredential instance for Azure authentication
        """
        # Cache the credentials object
        if Config.__azure_credentials is not None:
            return Config.__azure_credentials
        try:
            Config.__azure_credentials = DefaultAzureCredential()
            return Config.__azure_credentials
        except Exception as exc:
            logging.warning("Failed to create DefaultAzureCredential: %s", exc)
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
        except Exception as exc:
            logging.error("Failed to create CosmosDB client: %s. CosmosDB is required for this application.", exc)
            raise

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
        except Exception as exc:
            logging.error("Failed to get Azure OpenAI token: %s", exc)
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
    def GetAIProjectClient():
        """Create and return an AIProjectClient for Azure AI Foundry using from_connection_string.
        
        Returns:
            An AIProjectClient instance
        """
        if Config.__ai_project_client is not None:
            return Config.__ai_project_client
            
        try:
            credential = Config.GetAzureCredentials()
            if credential is None:
                raise RuntimeError("Unable to acquire Azure credentials; ensure DefaultAzureCredential is configured")
                
            connection_string = Config.AZURE_AI_AGENT_PROJECT_CONNECTION_STRING
            Config.__ai_project_client = AIProjectClient.from_connection_string(
                credential=credential,
                conn_str=connection_string
            )
            logging.info("Successfully created AIProjectClient using connection string")
            return Config.__ai_project_client
        except Exception as exc:
            logging.error("Failed to create AIProjectClient: %s", exc)
            raise
    
    @staticmethod
    async def CreateAzureAIAgent(
        kernel: Kernel, 
        agent_name: str, 
        instructions: str, 
        agent_type: str = "assistant", 
        tools=None,
        tool_resources=None,
        response_format=None,
        temperature: float = 0.0
    ):
        """
        Creates a new Azure AI Agent with the specified name and instructions using AIProjectClient.
        
        Args:
            kernel: The Semantic Kernel instance
            agent_name: The name of the agent
            instructions: The system message / instructions for the agent
            agent_type: The type of agent (defaults to "assistant")
            tools: Optional tool definitions for the agent
            tool_resources: Optional tool resources required by the tools
            response_format: Optional response format to control structured output
            temperature: The temperature setting for the agent (defaults to 0.0)
            
        Returns:
            A new AzureAIAgent instance
        """
        try:
            # Get the AIProjectClient
            project_client = Config.GetAIProjectClient()
            
            # Tool handling: We need to distinguish between our SK functions and
            # the tool definitions needed by project_client.agents.create_agent
            tool_definitions = None
            kernel_functions = []
            
            # If tools are provided and they are SK KernelFunctions, we need to handle them differently
            # than if they are already tool definitions expected by AIProjectClient
            if tools:
                # Check if tools are SK KernelFunctions
                if all(hasattr(tool, 'name') and hasattr(tool, 'invoke') for tool in tools):
                    # Store the kernel functions to register with the agent later
                    kernel_functions = tools
                    # For now, we don't extract tool definitions from kernel functions
                    # This would require additional code to convert SK functions to AI Project tool definitions
                    logging.warning("Kernel functions provided as tools will be registered with the agent after creation")
                else:
                    # Assume these are already proper tool definitions for create_agent
                    tool_definitions = tools
            
            # Create the agent using the project client
            logging.info("Creating agent '%s' with model '%s'", agent_name, Config.AZURE_OPENAI_DEPLOYMENT_NAME)
            agent_definition = await project_client.agents.create_agent(
                model=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
                name=agent_name,
                instructions=instructions,
                tools=tool_definitions,  # Only pass tool_definitions, not kernel functions
                tool_resources=tool_resources,
                temperature=temperature,
                response_format=response_format
            )
            
            # Create the agent instance directly with project_client and definition
            agent_kwargs = {
                "client": project_client,
                "definition": agent_definition,
                "kernel": kernel
            }
            
            agent = AzureAIAgent(**agent_kwargs)
            
            # Register the kernel functions with the agent if any were provided
            if kernel_functions:
                for function in kernel_functions:
                    if hasattr(agent, 'add_function'):
                        agent.add_function(function)
                    
            return agent
        except Exception as exc:
            logging.error("Failed to create Azure AI Agent: %s", exc)
            raise