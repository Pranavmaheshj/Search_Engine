from pathlib import Path
import re

def save_text_to_file(text: str, query: str, subfolder: str = "text_summaries") -> Path:
    """
    Saves a given string of text to a file and returns the path.
    
    Args:
        text: The text content to save.
        query: The original query or filename, used to create the output filename.
        subfolder: The name of the subfolder inside 'data' to save the file.
        
    Returns:
        A Path object to the newly created file.
    """
    # Define the main directory for saved summaries
    output_dir = Path("data") / subfolder
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a safe filename from the query
    safe_filename = re.sub(r'[^\w\-_\. ]', '_', query)[:50] + ".txt"
    filepath = output_dir / safe_filename
    
    # Write the summary to the file
    filepath.write_text(text, encoding='utf-8')
    
    return filepath