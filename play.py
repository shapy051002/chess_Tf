import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.datasets import make_regression
import chess
import chess.engine
import random
import numpy
import numpy as np
import math
import chess
import time
import chess.engine
import random

class ChessEvalNet(nn.Module):
    def __init__(self):
        super(ChessEvalNet, self).__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ChessEvalNet()

model.load_state_dict(torch.load('chess_eval_net.pth'))


model.eval()

print("Model loaded from chess_eval_net.pth")
def random_board(max_depth=200):
    while True:
        board = chess.Board()
        depth = random.randrange(0, max_depth)

        for _ in range(depth):
            all_moves = list(board.legal_moves)
            random_move = random.choice(all_moves)
            board.push(random_move)
            if board.is_game_over():
                break

        score = stockfish(board, 5)


        if score is not None and not math.isnan(score):
            return board

def stockfish(board, depth):
    with chess.engine.SimpleEngine.popen_uci(r"C:\Users\shrey\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe") as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        
        
        score = result['score'].white().score() if result['score'].white() is not None else 0.0

        return float(score) if score is not None else float('nan')

squares_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}
def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]

def split_dims(board):
    board3d = np.zeros((14,8,8), dtype=np.int8)
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8,8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8,8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux
    return board3d

def minimax_eval(board):
    board3d = split_dims(board)
    board3d = np.expand_dims(board3d, 0)
    board3d_tensor = torch.tensor(board3d, dtype=torch.float32)
    with torch.no_grad():
        output = model(board3d_tensor)
    return output.item()

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board)
    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_ai_move(board, depth):
    max_move = None
    max_eval = -np.inf
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -np.inf, np.inf, False)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            max_move = move
    return max_move 

def get_ai_move(board, depth):
    max_move = None
    max_eval = -np.inf
    

    maximizing_player = board.turn == chess.WHITE
    
    for move in board.legal_moves:
        board.push(move)
        
        
        eval = minimax_eval(board) if depth <= 1 else minimax(board, depth - 1, -np.inf, np.inf, maximizing_player)
        
        board.pop()
        
        if maximizing_player and eval > max_eval:
            max_eval = eval
            max_move = move
        elif not maximizing_player and eval < max_eval:
            max_eval = eval
            max_move = move
            
    return max_move
def generate_stockfish_move(board, depth):
    max_move = None
    max_eval = -np.inf
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -np.inf, np.inf, False)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            max_move = move
    return max_move 

board = chess.Board()
 
import sys
# Initialize the board
board = chess.Board()

while True:
    # Neural network's move
    move = get_ai_move(board, 1)
    board.push(move)
    print(f'\nNeural Network\'s move:\n{board}')
    if board.is_game_over():
        break
    
    # Stockfish's move
    stockfish_move = generate_stockfish_move(board, 3)
    board.push(stockfish_move)
    print(stockfish_move)
    print(f'\nStockfish\'s move:\n{board}')
    if board.is_game_over():
        break