
def format_time(seconds: float) -> str:
    """
    Format time duration into a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string:
        - If < 60s: "45.2s"
        - If >= 60s: "26m (1591.5s)"
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        return f"{minutes}m ({seconds:.2f}s)"
