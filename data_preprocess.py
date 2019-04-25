# exec(open('chess_data_preparator3.py').read())

import chess.pgn
import numpy as np
import tensorflow as tf


output = "train_dataset_2"
chess_games = "C:/Users/radu/Desktop/WIN19/COMP4106/final_project/chess_games/chess_games2.pgn"
print("Parsing games from" + chess_games)
print("Output file will be: " + output)

count_games = 0
samples = [] #array of chess positions
labels = [] #array of game results coresponding to chess position

pgn = open(chess_games)
results = {'0-1':0, '1-0':1} #0 - black wins, 1 - white wins

while True:
    # print(chess.pgn.read_game(pgn))
    count_games += 1
    chess_game = chess.pgn.read_game(pgn)
    if chess_game is None:
        break
    res = chess_game.headers["Result"]
    if res == "1-0" or res == "0-1":
        # print(chess_game.headers["Result"])
        # print(chess_game.headers["Round"])
        result = results[res]
        board = chess_game.board()
        count = 0
        pos_count = 0
        for move in chess_game.mainline_moves():
            count += 1
            if count > 7 and not board.is_capture(move) and not board.is_en_passant(move):
                pos_count += 1
                int_position = np.zeros(64, np.int8)
                for i in range(64):
                    piece = board.piece_at(i)
                    if piece is not None:
                        int_position[i] = {"P": 10, "N": 13, "B": 16, "R": 19, "Q": 22, "K": 25, \
                                     "p": 39, "n":42, "b":45, "r":48, "q":51, "k": 54}[piece.symbol()]
                if board.has_queenside_castling_rights(chess.WHITE):
                    int_position[0] = 17
                if board.has_kingside_castling_rights(chess.WHITE):
                    int_position[7] = 17
                if board.has_queenside_castling_rights(chess.BLACK):
                    int_position[56] = 57
                if board.has_kingside_castling_rights(chess.BLACK):
                    int_position[63] = 57
                
                if not board.turn:
                    int_position[:] = [x * -1 for x in int_position]

                int_position = int_position.reshape(8,8)
                samples.append(int_position)
                labels.append(result)
                # break
                if pos_count > 20:
                    break
            board.push(move)
    # break
    # if count_games == 1000:
        # break


print(len(samples))
print(len(labels))

samples = np.array(samples)
samples = tf.keras.utils.normalize(samples, axis=1)
labels = np.array(labels)

np.savez(output, samples, labels)

