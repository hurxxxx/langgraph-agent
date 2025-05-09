"""
DateTime Tool Module

This module provides utility functions for date and time operations,
which can be used by agents to get current date/time information,
calculate date differences, and format dates.
"""

import datetime
from typing import Optional, Union, Tuple
from langchain.tools import tool


@tool
def get_current_datetime(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date and time.
    
    Args:
        format_str: Format string for the datetime (default: "%Y-%m-%d %H:%M:%S")
        
    Returns:
        str: Current date and time formatted according to format_str
    """
    return datetime.datetime.now().strftime(format_str)


@tool
def get_current_date(format_str: str = "%Y-%m-%d") -> str:
    """
    Get the current date.
    
    Args:
        format_str: Format string for the date (default: "%Y-%m-%d")
        
    Returns:
        str: Current date formatted according to format_str
    """
    return datetime.datetime.now().strftime(format_str)


@tool
def format_date(date_str: str, input_format: str = "%Y-%m-%d", output_format: str = "%Y-%m-%d") -> str:
    """
    Format a date string from one format to another.
    
    Args:
        date_str: Date string to format
        input_format: Format of the input date string (default: "%Y-%m-%d")
        output_format: Format of the output date string (default: "%Y-%m-%d")
        
    Returns:
        str: Formatted date string
    """
    try:
        date_obj = datetime.datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
    except ValueError as e:
        return f"Error formatting date: {str(e)}"


@tool
def days_until(target_date_str: str, date_format: str = "%Y-%m-%d") -> int:
    """
    Calculate the number of days until a target date.
    
    Args:
        target_date_str: Target date string in the specified format
        date_format: Format of the date string (default: "%Y-%m-%d")
        
    Returns:
        int: Number of days until the target date (negative if in the past)
    """
    try:
        target_date = datetime.datetime.strptime(target_date_str, date_format).date()
        today = datetime.datetime.now().date()
        delta = target_date - today
        return delta.days
    except ValueError as e:
        return f"Error calculating days until: {str(e)}"


@tool
def days_since(past_date_str: str, date_format: str = "%Y-%m-%d") -> int:
    """
    Calculate the number of days since a past date.
    
    Args:
        past_date_str: Past date string in the specified format
        date_format: Format of the date string (default: "%Y-%m-%d")
        
    Returns:
        int: Number of days since the past date (negative if in the future)
    """
    try:
        past_date = datetime.datetime.strptime(past_date_str, date_format).date()
        today = datetime.datetime.now().date()
        delta = today - past_date
        return delta.days
    except ValueError as e:
        return f"Error calculating days since: {str(e)}"


@tool
def is_date_in_range(date_str: str, start_date_str: Optional[str] = None, 
                    end_date_str: Optional[str] = None, date_format: str = "%Y-%m-%d") -> bool:
    """
    Check if a date is within a specified range.
    
    Args:
        date_str: Date string to check
        start_date_str: Start date string (inclusive, optional)
        end_date_str: End date string (inclusive, optional)
        date_format: Format of the date strings (default: "%Y-%m-%d")
        
    Returns:
        bool: True if the date is within the range, False otherwise
    """
    try:
        date = datetime.datetime.strptime(date_str, date_format).date()
        
        if start_date_str:
            start_date = datetime.datetime.strptime(start_date_str, date_format).date()
            if date < start_date:
                return False
        
        if end_date_str:
            end_date = datetime.datetime.strptime(end_date_str, date_format).date()
            if date > end_date:
                return False
        
        return True
    except ValueError as e:
        return f"Error checking date range: {str(e)}"
