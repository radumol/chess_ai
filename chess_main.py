# exec(open('chess3.py').read())
#shortest mate: f2f3, e7e5, g2g4, d8h4

import chess.engine
import chess
from math import inf as infinity
from copy import deepcopy
import tensorflow as tf


from keras.models import load_model
# import chess.svg
# from IPython.display import SVG

import numpy as np

def data_preprocess(board2):
    wrapper = []
    int_position = np.zeros(64, np.int8)
    for i in range(64):
        piece = board2.piece_at(i)
        if piece is not None:
            int_position[i] = {"P": 10, "N": 13, "B": 16, "R": 19, "Q": 22, "K": 25, \
                         "p": 39, "n":42, "b":45, "r":48, "q":51, "k": 54}[piece.symbol()]
    if board2.has_queenside_castling_rights(chess.WHITE):
        int_position[0] = 17
    if board2.has_kingside_castling_rights(chess.WHITE):
        int_position[7] = 17
    if board2.has_queenside_castling_rights(chess.BLACK):
        int_position[56] = 57
    if board2.has_kingside_castling_rights(chess.BLACK):
        int_position[63] = 57
    
    # print(board2)
    
    if not board2.turn:
        int_position[:] = [x * -1 for x in int_position]
    # elif board2.turn:
        # int_position[:] = [x + 5 for x in int_position]
    
    int_position = int_position.reshape(8,8)
    wrapper.append(int_position)
    wrapper = np.array(wrapper)
    wrapper = tf.keras.utils.normalize(wrapper, axis=1)
    return wrapper

model = load_model('model_chess_final_2.h5')

class Node(object):

    def __init__(self, board):
        self.board = board
        self.evaluation = 0

    def heuristic(self):
        score = 0
        
        white_score = 0
        black_score = 0
        
        piece_values = {1:10, 2:30, 3:30, 4:50, 5:90, 6:0}
        white_material = 0
        black_material = 0
        
        #all squares attacked by a side:
        white_control = 0
        black_control = 0
        #number of stuck pieces of oposing side:
        white_pins = 0
        black_pins = 0
        
        for i in range(64):
            if self.board.piece_at(i) is not None:
                if self.board.piece_at(i).color:
                    white_material += piece_values[self.board.piece_at(i).piece_type]
                    white_control += self.board.attacks(i).__len__()
                    if self.board.is_pinned(chess.BLACK, i):
                        white_pins += 1
                else:
                    black_material += piece_values[self.board.piece_at(i).piece_type]
                    black_control += self.board.attacks(i).__len__()
                    if self.board.is_pinned(chess.WHITE, i):
                        black_pins += 1
        
        #control of central squares d4:27, d5:25, e4:28, e5:36 :
        white_central_control = 0
        white_central_control += self.board.attackers(chess.WHITE, 25).__len__()
        white_central_control += self.board.attackers(chess.WHITE, 27).__len__()
        white_central_control += self.board.attackers(chess.WHITE, 28).__len__()
        white_central_control += self.board.attackers(chess.WHITE, 36).__len__()
        black_central_control = 0
        black_central_control += self.board.attackers(chess.BLACK, 25).__len__()
        black_central_control += self.board.attackers(chess.BLACK, 27).__len__()
        black_central_control += self.board.attackers(chess.BLACK, 28).__len__()
        black_central_control += self.board.attackers(chess.BLACK, 36).__len__()
        
        #pawns placed on central squares d4:27, d5:25, e4:28, e5:36 :
        white_central_pawns = 0
        if str(self.board.piece_at(27)) == 'P':
            white_central_pawns += 2
        if str(self.board.piece_at(25)) == 'P':
            white_central_pawns += 1
        if str(self.board.piece_at(28)) == 'P':
            white_central_pawns += 2
        if str(self.board.piece_at(36)) == 'P':
            white_central_pawns += 1
        
        black_central_pawns = 0
        if str(self.board.piece_at(27)) == 'p':
            black_central_pawns += 1
        if str(self.board.piece_at(25)) == 'p':
            black_central_pawns += 2
        if str(self.board.piece_at(28)) == 'p':
            black_central_pawns += 1
        if str(self.board.piece_at(36)) == 'p':
            black_central_pawns += 2
        
        #piece mobility of each side:
        white_legal_moves = self.board.legal_moves.count()
        self.board.turn = not self.board.turn
        black_legal_moves = self.board.legal_moves.count()
        self.board.turn = not self.board.turn
        
        #queen safety:
        white_queen_safety = 0
        black_queen_safety = 0
        if len(self.board.move_stack) < 14:
            if len(list(self.board.pieces(chess.QUEEN, chess.WHITE))) > 0:
                if list(self.board.pieces(chess.QUEEN, chess.WHITE))[0] < 16:
                    white_queen_safety += 500
            if len(list(self.board.pieces(chess.QUEEN, chess.BLACK))) > 0:
                if list(self.board.pieces(chess.QUEEN, chess.BLACK))[0] > 47:
                    black_queen_safety += 500
        
        #king safety:
        white_king_safety = 0
        black_king_safety = 0
        queens = self.board.pieces(chess.QUEEN, chess.WHITE).__len__() + self.board.pieces(chess.QUEEN, chess.BLACK).__len__()
        if queens > 0:
            if self.board.king(chess.WHITE) < 8:
                white_king_safety += 20
            if self.board.king(chess.BLACK) > 55:
                black_king_safety += 20
        if self.board.is_check():
            if self.board.turn:
                white_king_safety -= 20
            else:
                black_king_safety -= 20
        if self.board.has_kingside_castling_rights(chess.WHITE):
            white_king_safety += 15
        if self.board.has_queenside_castling_rights(chess.WHITE):
            white_king_safety += 15
        if self.board.has_kingside_castling_rights(chess.BLACK):
            black_king_safety += 15
        if self.board.has_queenside_castling_rights(chess.BLACK):
            black_king_safety += 15
        if self.board.turn:
            if self.board.is_castling(self.board.move_stack[len(self.board.move_stack)-1]):
                black_king_safety += 1000
        else:
            if self.board.is_castling(self.board.move_stack[len(self.board.move_stack)-1]):
                white_king_safety += 1000
                
        
        white_material *= 40
        black_material *= 40
        white_control *= int(white_control*1.5)
        black_control *= int(black_control*1.5)
        white_pins *= 15
        black_pins *= 15
        white_legal_moves = int(white_legal_moves*1)
        black_legal_moves = int(black_legal_moves*1)
        white_central_control *= 2
        black_central_control *= 2
        white_king_safety *= 15
        black_king_safety *= 15
        white_central_pawns *= 5
        black_central_pawns *= 5
        white_queen_safety *= 2
        black_queen_safety *= 2
        
        white_score = white_material + white_control + white_pins + white_legal_moves + white_central_control + white_king_safety + white_central_pawns + white_queen_safety
        black_score = black_material + black_control + black_pins + black_legal_moves + black_central_control + black_king_safety + black_central_pawns + black_queen_safety
        score = white_score - black_score
        
        
        
        
        # print(self.board)
        # print(white_central_pawns)
        # print(score)
        # print()
        return score
    
    def heuristic2(self):
        score = []
        # print(self.board)
        parsed_position = data_preprocess(self.board)
        # print(parsed_position)
        # print()
        score = model.predict(parsed_position)
        # score = model.predict_classes(parsed_position)
        # print(score)
        final_score = score[0][1] - score[0][0]
        # print(final_score)
        return final_score
    
    def minimax(self, depth, alpha, beta, player, heuristic):
        if self.board.is_checkmate():
            if player:
               self.evaluation = -100000
            else:
                self.evaluation = 100000
            return self
        elif depth == 0:
            if heuristic == "classic":
                self.evaluation = self.heuristic()
            elif heuristic == "machine_learn":
                self.evaluation = self.heuristic2()
            # print(self.evaluation)
            # print(self.board)
            # print()
            return self
        
        potential_moves = []
        for move in self.board.legal_moves:
            potential_moves.append(move)
        
        if player:
            max_position = deepcopy(self)
            max_eval = -infinity
            max_position.evaluation = max_eval
            for move in potential_moves:
                new_position = deepcopy(self)
                new_position.board.push(move)
                new_position = new_position.minimax(depth-1, alpha, beta, False, heuristic)
                # print("wtf1")
                if(max_position.evaluation < new_position.evaluation):
                    max_position = new_position
                if PRUNING:
                    if alpha < new_position.evaluation:
                        alpha = new_position.evaluation
                    if alpha >= beta:
                        break
            return max_position
        else:
            min_position = deepcopy(self)
            min_eval = +infinity
            min_position.evaluation = min_eval
            for move in potential_moves:
                new_position = deepcopy(self)
                # print(new_position.board.turn)
                new_position.board.push(move)
                # print(new_position.board.turn)
                new_position = new_position.minimax(depth-1, alpha, beta, True, heuristic)
                # print("WTF2")
                if(min_position.evaluation > new_position.evaluation):
                    min_position = new_position
                if PRUNING:
                    if beta > new_position.evaluation:
                        beta = new_position.evaluation
                    if alpha >= beta:
                        break
            return min_position


PRUNING = True

color = {True:"WHITE", False:"BLACK"}
pieces = {1:"pawn", 2:"knight", 3:"bishop", 4:"rook", 5:"queen", 6:"king"}

board = chess.Board()

# SVG(chess.svg.board(board=board,size=400))

engine_stockfish10 = chess.engine.SimpleEngine.popen_uci("C:/Users/radu/Downloads/stockfish-10-win/stockfish-10-win/Windows/stockfish_10_x64.exe")
engine_komodo9 = chess.engine.SimpleEngine.popen_uci("C:/Users/radu/Downloads/komodo-9/komodo-9_9dd577/Windows/komodo-9.02-64bit.exe")

import chess.pgn
pgn_game = chess.pgn.Game()
pgn_game.headers["Event"] = "Example"
# node = pgn_game.add_variation(chess.Move.from_uci("e2e4"))
# node = node.add_variation(chess.Move.from_uci("e7e5"))
# node = node.add_variation(chess.Move.from_uci("f2f4"))
# node = node.add_variation(chess.Move.from_uci("f2f4"))

pgn_count = 0
node = pgn_game

game = Node(board)
best_position = game.minimax(3, -infinity, +infinity, True, "machine_learn")
best_move = best_position.board.move_stack[len(board.move_stack)]
print(best_move)
board.push(best_move)
node = pgn_game.add_variation(best_move)
while not board.is_game_over():
    print(board)
    print(color[board.turn] + " to move:")
    #human playing as WHITE:
    # if pgn_count == 4:
            # break
    if board.turn:
        
        game = Node(board)
        best_position = game.minimax(3, -infinity, +infinity, True, "machine_learn")
        best_move = best_position.board.move_stack[len(board.move_stack)]
        board.push(best_move)
        print(best_position.evaluation)
        
        
        # result = engine_stockfish10.play(board, chess.engine.Limit(time=0.100))
        # best_move = result.move
        # board.push(result.move)
        
        # move = input()
        # if move == "quit":
            # break
        # best_move = chess.Move.from_uci(move)
        # print(best_move)
        # if best_move in board.legal_moves:
            # board.push(best_move)
        # else:
            # print("Move not legal WTF")
        
        node = node.add_variation(best_move)
            
        pgn_count += 1
        
    #AI playing as BLACK:
    else:
        result = engine_komodo9.play(board, chess.engine.Limit(time=0.500))
        best_move = result.move
        board.push(best_move)
        
        # game = Node(board)
        # best_position = game.minimax(2, -infinity, +infinity, False, "machine_learn")
        # best_move = best_position.board.move_stack[len(board.move_stack)]
        # board.push(best_move)
        
        node = node.add_variation(best_move)
        
        pgn_count += 1
    print(best_move)
    print("Move #" + str(pgn_count))



print(board)
print(board.move_stack)
print(len(board.move_stack))
if board.is_checkmate():
    print(color[not board.turn] + " WON!!! ")
else:
    print("Game ended in a DRAW!")

text_file = open("game_pgn.txt", "w")
text_file.write(str(pgn_game))
text_file.close()