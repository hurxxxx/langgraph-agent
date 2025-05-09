"""
LangGraph DateTime Integration

This module provides LangGraph integration for date and time operations,
allowing date/time tools to be used as nodes in a LangGraph workflow.
"""

import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple, TypedDict, Annotated
import operator

from langchain.tools import tool
from langgraph.graph import StateGraph, END


class DateTimeState(TypedDict):
    """State for the DateTime node."""
    query: str
    current_date: Optional[str]
    current_time: Optional[str]
    current_datetime: Optional[str]
    days_until_result: Optional[int]
    days_since_result: Optional[int]
    is_recent_result: Optional[bool]
    formatted_date_result: Optional[str]
    error: Optional[str]

    # Parameters for operations
    target_date: Optional[str]
    past_date: Optional[str]
    check_date: Optional[str]
    format_date: Optional[str]
    input_format: Optional[str]
    output_format: Optional[str]
    date_format: Optional[str]
    days: Optional[int]


def get_current_date_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the current date and add it to the state.

    Args:
        state: Current state

    Returns:
        Dict: Updated state with current date
    """
    try:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        return {"current_date": current_date}
    except Exception as e:
        return {"error": f"Error getting current date: {str(e)}"}


def get_current_time_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the current time and add it to the state.

    Args:
        state: Current state

    Returns:
        Dict: Updated state with current time
    """
    try:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return {"current_time": current_time}
    except Exception as e:
        return {"error": f"Error getting current time: {str(e)}"}


def get_current_datetime_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the current date and time and add it to the state.

    Args:
        state: Current state

    Returns:
        Dict: Updated state with current date and time
    """
    try:
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {"current_datetime": current_datetime}
    except Exception as e:
        return {"error": f"Error getting current datetime: {str(e)}"}


def days_until_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate days until a target date and add it to the state.

    Args:
        state: Current state with target_date

    Returns:
        Dict: Updated state with days_until_result
    """
    try:
        target_date = state.get("target_date")
        if not target_date:
            return {"error": "No target date provided"}

        date_format = state.get("date_format", "%Y-%m-%d")
        target_date_obj = datetime.datetime.strptime(target_date, date_format).date()
        today = datetime.datetime.now().date()
        delta = target_date_obj - today

        return {"days_until_result": delta.days}
    except Exception as e:
        return {"error": f"Error calculating days until: {str(e)}"}


def days_since_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate days since a past date and add it to the state.

    Args:
        state: Current state with past_date

    Returns:
        Dict: Updated state with days_since_result
    """
    try:
        past_date = state.get("past_date")
        if not past_date:
            return {"error": "No past date provided"}

        date_format = state.get("date_format", "%Y-%m-%d")
        past_date_obj = datetime.datetime.strptime(past_date, date_format).date()
        today = datetime.datetime.now().date()
        delta = today - past_date_obj

        return {"days_since_result": delta.days}
    except Exception as e:
        return {"error": f"Error calculating days since: {str(e)}"}


def is_recent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if a date is recent and add the result to the state.

    Args:
        state: Current state with check_date and days

    Returns:
        Dict: Updated state with is_recent_result
    """
    try:
        date_str = state.get("check_date")
        if not date_str:
            return {"error": "No date provided"}

        days = state.get("days", 7)  # Default to 7 days
        date_format = state.get("date_format", "%Y-%m-%d")

        date_obj = datetime.datetime.strptime(date_str, date_format).date()
        today = datetime.datetime.now().date()
        delta = abs((today - date_obj).days)

        return {"is_recent_result": delta <= days}
    except Exception as e:
        return {"error": f"Error checking if date is recent: {str(e)}"}


def format_date_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a date and add it to the state.

    Args:
        state: Current state with format_date, input_format, and output_format

    Returns:
        Dict: Updated state with formatted_date_result
    """
    try:
        date_str = state.get("format_date")
        if not date_str:
            return {"error": "No date provided"}

        input_format = state.get("input_format", "%Y-%m-%d")
        output_format = state.get("output_format", "%Y-%m-%d")

        date_obj = datetime.datetime.strptime(date_str, input_format)
        formatted_date = date_obj.strftime(output_format)

        return {"formatted_date_result": formatted_date}
    except Exception as e:
        return {"error": f"Error formatting date: {str(e)}"}


def create_datetime_graph() -> StateGraph:
    """
    Create a LangGraph for date and time operations.

    Returns:
        StateGraph: LangGraph for date and time operations
    """
    # Create the graph
    workflow = StateGraph(DateTimeState)

    # Add nodes
    workflow.add_node("get_current_date", get_current_date_node)
    workflow.add_node("get_current_time", get_current_time_node)
    workflow.add_node("get_current_datetime", get_current_datetime_node)
    workflow.add_node("days_until", days_until_node)
    workflow.add_node("days_since", days_since_node)
    workflow.add_node("is_recent", is_recent_node)
    workflow.add_node("format_date", format_date_node)

    # Set entry point
    workflow.set_entry_point("get_current_datetime")

    # Add conditional edges based on the query
    def route_by_query(state: Dict[str, Any]) -> str:
        query = state.get("query", "").lower()

        if "current date" in query or "today" in query:
            return "get_current_date"
        elif "current time" in query or "now" in query:
            return "get_current_time"
        elif "current datetime" in query:
            return "get_current_datetime"
        elif "days until" in query:
            return "days_until"
        elif "days since" in query:
            return "days_since"
        elif "is recent" in query:
            return "is_recent"
        elif "format date" in query:
            return "format_date"
        else:
            return "get_current_datetime"  # Default

    # Add conditional edges
    workflow.add_conditional_edges(
        "get_current_datetime",
        route_by_query,
        {
            "get_current_date": "get_current_date",
            "get_current_time": "get_current_time",
            "get_current_datetime": END,
            "days_until": "days_until",
            "days_since": "days_since",
            "is_recent": "is_recent",
            "format_date": "format_date"
        }
    )

    # Add edges to END
    workflow.add_edge("get_current_date", END)
    workflow.add_edge("get_current_time", END)
    workflow.add_edge("days_until", END)
    workflow.add_edge("days_since", END)
    workflow.add_edge("is_recent", END)
    workflow.add_edge("format_date", END)

    return workflow
