import sys
import re

def remove_line_numbers(input_text):
    """
    Removes leading line numbers and a pipe character from each line of the input text.

    This function processes a multi-line string, where each line may start with a number
    followed by a pipe character (`|`). It removes these prefixes and returns the cleaned
    text.

    Args:
        input_text (str): The input text containing lines with optional leading numbers
                          and pipe characters.

    Returns:
        str: The cleaned text with line numbers and pipe characters removed.

    Example:
        >>> input_text = "1| First line\n2| Second line\n3| Third line"
        >>> remove_line_numbers(input_text)
        'First line\nSecond line\nThird line'
    """
    cleaned_lines = []
    for line in input_text.splitlines():
        # Use regex to remove leading numbers and pipe character
        cleaned_line = re.sub(r'^\s*\d+\|\s*', '', line)
        cleaned_lines.append(cleaned_line)
    return '\n'.join(cleaned_lines)

if __name__ == "__main__":
    # Read from standard input or you can modify to read from a file
    input_text = sys.stdin.read()
    output_text = remove_line_numbers(input_text)
    print(output_text)
