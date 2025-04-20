import sys
import os
import pytest
import logging

# Ensure src/backend is on the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_kernel import Config

# Configure logging for the tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_azure_project_client_connection():
    """
    Integration test to verify that we can successfully create a connection to Azure using the project client.
    This is the most basic test to ensure our Azure connectivity is working properly.
    """
    try:
        # Get the Azure AI Project client
        project_client = Config.GetAIProjectClient()
        
        # Verify the project client has been created successfully
        assert project_client is not None, "Failed to create Azure AI Project client"
        
        # Check that the connection string environment variable is set
        conn_str_env = os.environ.get("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING")
        assert conn_str_env is not None, "AZURE_AI_AGENT_PROJECT_CONNECTION_STRING environment variable not set"
        
        # Log success
        logger.info("Successfully connected to Azure using the project client")
        
        # Return client for reference (though not needed for test)
        return project_client
        
    except Exception as e:
        logger.error(f"Error connecting to Azure: {str(e)}")
        raise
