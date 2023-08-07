import sys

def split_lines(file_path, word, output_file_path, none_file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filter out lines containing the word 'None'
    lines_with_word = [line for line in lines if word in line]
    lines_without_word = [line for line in lines if word not in line]

    with open(output_file_path, 'w') as output_file:
        output_file.writelines(lines_without_word)

    with open(none_file_path, 'w') as none_file:
        none_file.writelines(lines_with_word)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script_name.py <input_file> <word_to_remove> <output_file> <none_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    word_to_remove = sys.argv[2]
    output_file_path = sys.argv[3]
    none_file_path = sys.argv[4]

    split_lines(input_file_path, word_to_remove, output_file_path, none_file_path)
