from pathlib import Path
import chess.pgn
import re

file_to_open = "lichess_db_1.pgn"

database = open(file_to_open)

first_game = chess.pgn.read_game(database)

print(first_game.headers["Event"])
print(first_game.headers["White"])
print(first_game.headers["Black"])


# Regular expression to match evaluation comments
eval_regex = r'\[%eval\s+([-+]?\d+\.\d+)\]'

board = first_game.board()

for move in first_game.mainline_moves():
    board.push(move)

    # Get the comment for the current move
    comment = first_game.board().san(move)

    # Extract the evaluation from the comment using regex
    eval_match = re.search(eval_regex, comment)

    if eval_match:
        evaluation = eval_match.group(1)
        print(f"Move: {move}, Evaluation: {evaluation}")
    else:
        print(f"Move: {move}, No evaluation found.")


# board = first_game.board()
# for move in first_game.mainline_moves():
# 	board.push(move)
# 	print(move.uci())
# 	print(first_game.comment)
# # print(first_game.comment)



# # Iterate through all moves and play them on a board.
# board = first_game.board()
# for move in first_game.mainline_moves():
# board.push(move)

# board