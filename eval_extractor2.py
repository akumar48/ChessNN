import argparse
import chess.pgn

def game_has_eval_tag(game):
    node = game
    while node:
        if 'eval' in node.comment:
            return True
        node = node.variations[0] if node.variations else None
    return False

def filter_games(input_file, output_file):
    with open(input_file, 'r') as f:
        with open(output_file, 'w') as out_file:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                if game_has_eval_tag(game):
                    out_file.write(str(game) + '\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter PGN games based on the presence of 'eval' tag.")
    parser.add_argument("input_file", help="Input PGN file name")
    parser.add_argument("output_file", help="Output PGN file name")
    args = parser.parse_args()

    filter_games(args.input_file, args.output_file)
    print("Filtered games have been saved to", args.output_file)
