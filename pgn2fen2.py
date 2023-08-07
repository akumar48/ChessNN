import chess.pgn
import subprocess
import sys
import time

TIME_PER_MOVE = .2  # seconds



def get_lc0_evaluation_and_best_move(fen_position):
    lc0_process = subprocess.Popen(['stockfish/stockfish-windows-x86-64-avx2.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    command = f'position fen {fen_position}\ngo\n'
    
    lc0_process.stdin.write(command)
    lc0_process.stdin.flush()
    time.sleep(TIME_PER_MOVE)
    lc0_process.stdin.write('stop\n')
    lc0_process.stdin.flush()
    
    output, _ = lc0_process.communicate()
    lines = reversed(output.strip().split('\n'))
    evaluation = None
    best_move = None

    for line in lines:
        if 'bestmove' in line:
            # print(f'bestline:{line}')
            parts = line.split()
            if len(parts) >= 2:
                if len(parts[1]) == 4 or len(parts[1]) == 5:
                    best_move = parts[1]
                    print(f'best_move: {best_move}')
        elif 'cp' in line:
            if evaluation is None:
                parts = line.split('cp')
                clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
                evaluation = clamp(int(parts[1].split()[0]), -750, 750)
                print(parts[0].split('seldepth')[0])
                print(f'evaluation: {evaluation}')
        elif 'mate' in line:
            if evaluation is None:
                parts = line.split('mate')
                mate_in = int(parts[1].split()[0])
                who_mating = 0 if mate_in==0 else 1 if mate_in>0 else -1
                evaluation = who_mating * int((300/mate_in)+750)
                print(parts[0].split('seldepth')[0])
                print(f'mate   eval: {evaluation}')   

    return best_move, evaluation

def pgn_to_fen_with_evaluation(input_pgn_file, output_file_path):
    with open(input_pgn_file) as pgn_file, open(output_file_path, 'w') as output_file:
        game = chess.pgn.read_game(pgn_file)
        while game is not None:
            print("Processing game:", game.headers.get("White", "Black"))
            board = game.board()

            moves = list(game.mainline_moves())
            for i in range(len(moves) - 1):  # Iterate until the second-to-last move
                move = moves[i]
                board.push(move)
                fen_position = board.fen()
                print("Evaluating FEN:", fen_position)
                best_move, evaluation = get_lc0_evaluation_and_best_move(fen_position)
                if best_move is not None:
                    output_file.write(f"{fen_position} {evaluation} {best_move}\n")
                    output_file.flush()  # Flush the output buffer to write to the file immediately
                    print(f"{fen_position} {evaluation} {best_move}\n")
                else:
                    print(f'best_move:{best_move}')

            game = chess.pgn.read_game(pgn_file)
            # print("Finished game:", game.headers.get("Event", "Unknown Event"), "\n")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_pgn_file output_file")
        sys.exit(1)

    input_pgn_file = sys.argv[1]
    output_file = sys.argv[2]
    pgn_to_fen_with_evaluation(input_pgn_file, output_file)
