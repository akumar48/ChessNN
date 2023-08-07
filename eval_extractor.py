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
    input_file = "lichess_db_2.pgn"
    output_file = "eval_games_2.pgn"
    filter_games(input_file, output_file)
    print("Filtered games have been saved to", output_file)
