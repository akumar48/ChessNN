import chess.pgn
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

TIME_PER_MOVE = 10  # seconds

def run_lc0_process(fen_position, result_queue):
    lc0_process = subprocess.Popen(['Lc0/lc0.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
                if len(parts[1]) == 4:
                    best_move = parts[1]
                    print(f'best_move: {best_move}')
        elif 'cp' in line:
            if evaluation is None:
                parts = line.split('cp')
                evaluation = int(parts[1].split()[0])
                print(f'evaluation: {evaluation}')

    result_queue.put((best_move, evaluation, fen_position))

def process_game_batch(game_batch, result_queue):
    for game in game_batch:
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            fen_position = board.fen()
            run_lc0_process(fen_position, result_queue)


def pgn_to_fen_with_evaluation(input_pgn_file, output_file_path, batch_size=10, max_queue_size=50):
    with open(input_pgn_file) as pgn_file:
        games = list(chess.pgn.read_game(pgn_file))

    print("queueing")
    result_queue = Queue(maxsize=max_queue_size)  # Using a fixed-size queue

    print("pooling")
    with ThreadPoolExecutor() as executor:
        batched_games = [games[i:i + batch_size] for i in range(0, len(games), batch_size)]
        for game_batch in batched_games:
            executor.submit(process_game_batch, game_batch, result_queue)

    print("starting output")
    with open(output_file_path, 'w') as output_file:
        with ThreadPoolExecutor() as write_executor:  # ThreadPoolExecutor for concurrent writing
            while not result_queue.empty() or write_executor._work_queue.qsize() > 0:
                try:
                    best_move, evaluation, fen_position = result_queue.get(timeout=1)
                    if best_move is not None:
                        write_executor.submit(output_file.write, f"{fen_position} {evaluation} {best_move}\n")
                except Empty:
                    pass
                output_file.flush()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_pgn_file output_file")
        sys.exit(1)

    input_pgn_file = sys.argv[1]
    output_file = sys.argv[2]
    pgn_to_fen_with_evaluation(input_pgn_file, output_file)
