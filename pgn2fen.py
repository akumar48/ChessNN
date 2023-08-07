import argparse
import chess.pgn
import re

def extract_moves(game):
    moves = []
    node = game
    while node.variations:
        next_node = node.variations[0]
        move = next_node.move
        moves.append(move)
        node = next_node
    return moves

def extract_evaluations(game):
    evaluations = []
    node = game
    while node.variations:
        next_node = node.variations[0]
        eval_str = next_node.comment.strip()
        eval_value = re.search(r"\b-?\d+\.?\d*\b", eval_str)
        if eval_value:
            evaluations.append(eval_value.group())
        else:
            evaluations.append("N/A")
        node = next_node
    return evaluations

def generate_fen_with_evals(input_file, output_file):
    with open(input_file, 'r') as f:
        with open(output_file, 'w') as out_file:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                moves = extract_moves(game)
                evaluations = extract_evaluations(game)
                fen_board = chess.Board()
                for move, eval_str in zip(moves, evaluations):
                    fen = fen_board.fen()
                    out_file.write(f"{fen} {eval_str}\n")
                    fen_board.push(move)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract FEN and evaluations from PGN games.")
    parser.add_argument("input_file", help="Input PGN file name")
    parser.add_argument("output_file", help="Output file name for FEN and evaluations")
    args = parser.parse_args()

    generate_fen_with_evals(args.input_file, args.output_file)
    print("FEN and evaluation data have been saved to", args.output_file)
