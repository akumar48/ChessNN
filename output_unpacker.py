import sys

def find_index_of_one(arr):
    if 1 in arr:
        return arr.index(1)
    else:
        return -1  # Return -1 if the element '1' is not found in the array

def number_to_lowercase_letter(array):
	num = find_index_of_one(array)
    if 1 <= num <= 26:
        letter = chr(ord('a') + num - 1)
        return letter.lower()
    else:
        return None  # Return None for numbers outside the range of valid lowercase letters

def output_to_eval(output):
	return 1050*output[0]

def output_to_move(output):
	move = ''
	first_letter = output[1:8]
	first_number = output[9:16]
	i_f_n = find_index_of_one(first_number)
	second_letter = output[17:24]
	second_number = output[25:32]
	i_s_n = find_index_of_one(second_number)
	promotion_letter = outptut[33:36]

	move += number_to_lowercase_letter(first_letter)
	if i_f_n != -1:
		move += str(i_f_n + 1)
	move += number_to_lowercase_letter(second_letter)
	if i_s_n != -1:
		move += str(i_s_n + 1)
	move += number_to_lowercase_letter(promotion_letter)