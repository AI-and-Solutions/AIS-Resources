import pygame
import torch
import numpy as np
from NNTicTacToe import TicTacToeNN

# Initialize PyGame
pygame.init()

# Constants
WIDTH, HEIGHT = 300, 300
LINE_WIDTH = 5
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

# PyGame screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe")
screen.fill(BG_COLOR)

# Board
board = np.zeros((BOARD_ROWS, BOARD_COLS))  # 0 = empty, 1 = player, -1 = AI

# Model
model = TicTacToeNN()
model.load_state_dict(torch.load("model.pth"))  # Ensure your trained model file is here
model.eval()

def draw_lines():
    """Draw the grid lines on the board."""
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, LINE_COLOR, (0, row * SQUARE_SIZE), (WIDTH, row * SQUARE_SIZE), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, LINE_COLOR, (col * SQUARE_SIZE, 0), (col * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

def draw_figures():
    """Draw the current board state."""
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row, col] == 1:
                pygame.draw.circle(screen, CIRCLE_COLOR, (int(col * SQUARE_SIZE + SQUARE_SIZE // 2), 
                                                          int(row * SQUARE_SIZE + SQUARE_SIZE // 2)), CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row, col] == -1:
                pygame.draw.line(screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE), 
                                 (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), 
                                 (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)

def mark_square(row, col, player):
    """Mark a square on the board."""
    board[row, col] = player

def available_square(row, col):
    """Check if a square is available."""
    return board[row, col] == 0

def is_board_full():
    """Check if the board is full."""
    return not (board == 0).any()

def check_winner(player):
    """Check if a player has won."""
    for row in range(BOARD_ROWS):
        if np.all(board[row, :] == player):
            return True
    for col in range(BOARD_COLS):
        if np.all(board[:, col] == player):
            return True
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

def ai_move():
    """Determine the AI's move using the trained model."""
    board_flat = board.flatten()
    board_tensor = torch.tensor(board_flat, dtype=torch.float32).unsqueeze(0)  # Batch of 1
    with torch.no_grad():
        predictions = model(board_tensor)
    move = torch.argmax(predictions).item()
    row, col = divmod(move, BOARD_COLS)
    while not available_square(row, col):  # Find a valid move
        predictions[0, move] = -float('inf')  # Penalize invalid moves
        move = torch.argmax(predictions).item()
        row, col = divmod(move, BOARD_COLS)
    return row, col

# Game loop
draw_lines()
player = 1
game_over = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            mouseX, mouseY = event.pos
            clicked_row, clicked_col = mouseY // SQUARE_SIZE, mouseX // SQUARE_SIZE

            if available_square(clicked_row, clicked_col):
                mark_square(clicked_row, clicked_col, player)
                draw_figures()

                if check_winner(player):
                    print("Player wins!" if player == 1 else "AI wins!")
                    game_over = True
                elif is_board_full():
                    print("It's a draw!")
                    game_over = True
                else:
                    player *= -1  # Switch player

        if player == -1 and not game_over:  # AI's turn
            row, col = ai_move()
            mark_square(row, col, player)
            draw_figures()

            if check_winner(player):
                print("Player wins!" if player == 1 else "AI wins!")
                game_over = True
            elif is_board_full():
                print("It's a draw!")
                game_over = True
            else:
                player *= -1  # Switch player

    pygame.display.update()
