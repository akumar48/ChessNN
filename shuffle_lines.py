import sys
import random

def shuffle_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    with open(file_path, 'w') as file:
        file.writelines(lines)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]

    shuffle_lines(file_path)
