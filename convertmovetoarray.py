def convert_move_to_array(move):
    # Assuming the move is in the format "e2e4"
    start_square, end_square = move[:2], move[2:]
    start_rank = ord(start_square[0]) - ord('a')  # Convert file to index (0 to 7)
    print(start_rank)
    start_file = int(start_square[1]) - 1  # Convert rank to index (0 to 7)
    print(start_file)
    
    end_rank = ord(end_square[0]) - ord('a')  # Convert file to index (0 to 7)
    print(end_rank)
    end_file = int(end_square[1]) - 1  # Convert rank to index (0 to 7)
    print(end_file)


    array = [0] * 32
    array[start_rank] = 1
    array[8 + start_file] = 1
    array[16 + end_rank] = 1
    array[24 + end_file] = 1

    return array

print(convert_move_to_array("a1h8"))