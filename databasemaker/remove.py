import sys
import re

def remove_line_numbers(input_text):
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
