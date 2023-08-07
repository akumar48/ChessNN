import sys

def convert_move_to_array(move):
    array = [0] * 36


    # Assuming the move is in the format "e2e4"
    start_square, end_square = move[:2], move[2:]
    array[ord(start_square[0]) - ord('a')] = 1
    array[7 + int(start_square[1])] = 1
    array[16 + ord(end_square[0]) - ord('a')] = 1
    array[23 + int(end_square[1])] = 1

    if len(move) == 5:
        promotion = end_square[2]
        if promotion == 'q':
            array[32] = 1
        elif promotion == 'r':
            array[33] = 1
        elif promotion == 'n':
            array[34] = 1
        elif promotion == 'b':
            array[35] = 1

    return array
    
def fen_to_numeric(fen):
    piece_values = {'K': 1.0, 'Q': 0.6, 'R': 0.33, 'B': 0.27, 'N': 0.2, 'P': 0.067,
                    'k': -1.0, 'q': -0.6, 'r': -0.33, 'b': -0.27, 'n': -0.2, 'p': -0.067}

    board, active_color, castling_rights, en_passant, halfmove_clock, fullmove_number = fen.split(' ')
    board_rows = board.split('/')
    numeric_board = []

    for row in board_rows:
        numeric_row = []
        for char in row:
            if char.isdigit():
                numeric_row.extend([0.0] * int(char))
            else:
                numeric_row.append(piece_values[char])
        numeric_board.extend(numeric_row)

    if active_color == 'w':
        numeric_board.append(1)
    elif active_color == 'b':
        numeric_board.append(0)

    castling_array = [0, 0, 0, 0]
    for letter in castling_rights:
        if letter == 'K':
            castling_array[0] = 1
        elif letter == 'Q':
            castling_array[1] = 1
        elif letter == 'k':
            castling_array[2] = 1
        elif letter == 'q':
            castling_array[3] = 1
    numeric_board.extend(castling_array)

    # print(f'en_passant: {en_passant}')
    en_passant_array = [0]*16
    if en_passant != '-':
        en_passant_array[ord(en_passant[0]) - ord('a')] = 1
        en_passant_array[int(en_passant[1]) - 1] = 1
    numeric_board.extend(en_passant_array)

    print(f'input dimension : {len(numeric_board)}')
    print('output dimension: 37')
    return numeric_board


# Check if the correct number of arguments are provided
if len(sys.argv) != 3:
    print("Usage: python script.py input_file.txt output_file.txt")
    sys.exit(1)

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

positions = []
evaluations = []
best_moves = []

with open(input_file_path, "r") as file:
    for line in file:
        # print(f'line: {line}')
        data = line.strip().split()
        print(f'data: {data}')
        fen = " ".join(data[:-2])
        # print(f'fen: {fen}')
        evaluation = int(data[-2]) if data[-2] else 0  # Assign 0 if evaluation is missing
        normalized_evaluation = round(evaluation / 1050, 4)
        best_move = data[-1]
        best_move_array = convert_move_to_array(best_move)
        positions.append(fen_to_numeric(fen))
        evaluations.append(normalized_evaluation)
        best_moves.append(best_move_array)

# Now you have the positions, evaluations, and best moves in separate lists.

# Write the processed data to the output file
with open(output_file_path, "w") as output_file:
    for position, evaluation, best_move in zip(positions, evaluations, best_moves):
        position_str = ",".join(str(piece) for piece in position)
        evaluation_str = str(evaluation)
        best_move_str = ",".join(str(move) for move in best_move)
        output_file.write(f"{position_str} {evaluation_str},{best_move_str}\n")
