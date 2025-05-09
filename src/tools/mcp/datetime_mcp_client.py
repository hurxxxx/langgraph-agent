"""
DateTime MCP Client

This module implements a client for the DateTime MCP server, which provides
date and time operations for agents in the system.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple

# Import MCP client components
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
except ImportError:
    # Mock implementation for when MCP is not available
    class ClientSession:
        def __init__(self, *args, **kwargs):
            pass
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            pass
        
        async def initialize(self):
            pass
    
    class StdioServerParameters:
        def __init__(self, *args, **kwargs):
            pass
    
    async def stdio_client(*args, **kwargs):
        class MockClient:
            async def __aenter__(self):
                return (None, None)
            
            async def __aexit__(self, *args):
                pass
        
        return MockClient()
    
    async def load_mcp_tools(*args, **kwargs):
        return []


class DateTimeMCPClient:
    """
    Client for the DateTime MCP server.
    """
    
    def __init__(self, server_script_path: Optional[str] = None):
        """
        Initialize the DateTime MCP client.
        
        Args:
            server_script_path: Path to the server script (default: None, uses package path)
        """
        if server_script_path is None:
            # Use the package path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.server_script_path = os.path.join(current_dir, "datetime_mcp_server.py")
        else:
            self.server_script_path = server_script_path
        
        self.tools = []
    
    async def initialize(self) -> List[Any]:
        """
        Initialize the client and load the tools.
        
        Returns:
            List[Any]: List of loaded tools
        """
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path]
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.tools = await load_mcp_tools(session)
                    return self.tools
        except Exception as e:
            print(f"Error initializing DateTime MCP client: {str(e)}")
            return []
    
    def get_tools(self) -> List[Any]:
        """
        Get the loaded tools.
        
        Returns:
            List[Any]: List of loaded tools
        """
        return self.tools
    
    @staticmethod
    def create_client() -> 'DateTimeMCPClient':
        """
        Create and initialize a DateTime MCP client.
        
        Returns:
            DateTimeMCPClient: Initialized client
        """
        client = DateTimeMCPClient()
        
        # Initialize the client
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(client.initialize())
        
        return client
