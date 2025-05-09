"""
DateTime MCP Server

This module implements a Model Context Protocol (MCP) server for date and time operations.
It exposes tools for getting the current date/time, formatting dates, and calculating
date differences.
"""

import datetime
from typing import Optional, Union, Dict, Any

# Import MCP server components
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # Mock implementation for when MCP is not available
    class FastMCP:
        def __init__(self, name):
            self.name = name
            self.tools = []
        
        def tool(self, *args, **kwargs):
            def decorator(func):
                self.tools.append(func)
                return func
            return decorator
        
        def run(self, **kwargs):
            print(f"[Mock] Running MCP server: {self.name}")


# Create MCP server
mcp = FastMCP("DateTimeTools")


@mcp.tool()
def current_datetime(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current date and time.
    
    Args:
        format_str: Format string for the datetime (default: "%Y-%m-%d %H:%M:%S")
    """
    return datetime.datetime.now().strftime(format_str)


@mcp.tool()
def current_date(format_str: str = "%Y-%m-%d") -> str:
    """
    Returns the current date.
    
    Args:
        format_str: Format string for the date (default: "%Y-%m-%d")
    """
    return datetime.datetime.now().strftime(format_str)


@mcp.tool()
def format_date(date_str: str, input_format: str = "%Y-%m-%d", output_format: str = "%Y-%m-%d") -> str:
    """
    Formats a date string from one format to another.
    
    Args:
        date_str: Date string to format
        input_format: Format of the input date string (default: "%Y-%m-%d")
        output_format: Format of the output date string (default: "%Y-%m-%d")
    """
    try:
        date_obj = datetime.datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
    except ValueError as e:
        return f"Error formatting date: {str(e)}"


@mcp.tool()
def days_until(date_str: str, date_format: str = "%Y-%m-%d") -> int:
    """
    Returns the number of days from today until a future date.
    
    Args:
        date_str: A date in the specified format
        date_format: Format of the date string (default: "%Y-%m-%d")
    """
    try:
        target_date = datetime.datetime.strptime(date_str, date_format).date()
        delta = target_date - datetime.datetime.now().date()
        return max(delta.days, 0)
    except ValueError as e:
        return f"Error calculating days until: {str(e)}"


@mcp.tool()
def days_since(date_str: str, date_format: str = "%Y-%m-%d") -> int:
    """
    Returns the number of days from a past date until today.
    
    Args:
        date_str: A date in the specified format
        date_format: Format of the date string (default: "%Y-%m-%d")
    """
    try:
        past_date = datetime.datetime.strptime(date_str, date_format).date()
        delta = datetime.datetime.now().date() - past_date
        return max(delta.days, 0)
    except ValueError as e:
        return f"Error calculating days since: {str(e)}"


@mcp.tool()
def is_recent(date_str: str, days: int = 7, date_format: str = "%Y-%m-%d") -> bool:
    """
    Checks if a date is within the specified number of days from today.
    
    Args:
        date_str: A date in the specified format
        days: Number of days to consider as recent (default: 7)
        date_format: Format of the date string (default: "%Y-%m-%d")
    """
    try:
        date = datetime.datetime.strptime(date_str, date_format).date()
        delta = datetime.datetime.now().date() - date
        return abs(delta.days) <= days
    except ValueError as e:
        return f"Error checking if date is recent: {str(e)}"


# Run the server if this file is executed directly
if __name__ == "__main__":
    mcp.run(transport="stdio")
