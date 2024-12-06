class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:
        # Function to check if player with mark 'X' or 'O' has won
        def has_winner(mark):
            # Check for horizontal and vertical wins
            for i in range(3):
                if all(board[i][j] == mark for j in range(3)):
                    return True
                if all(board[j][i] == mark for j in range(3)):
                    return True
          
            # Check for the two diagonal wins
            if all(board[i][i] == mark for i in range(3)):
                return True
            return all(board[i][2 - i] == mark for i in range(3))

        # Count the number of 'X's and 'O's on the board
        count_x = sum(row.count('X') for row in board)
        count_o = sum(row.count('O') for row in board)

        # Check for the correct number of 'X's and 'O's
        if count_x != count_o and count_x - 1 != count_o:
            return False

        # If 'X' has won, 'X' must be one more than 'O'
        if has_winner('X') and count_x - 1 != count_o:
            return False

        # If 'O' has won, the count of 'X' and 'O' must be the same
        if has_winner('O') and count_x != count_o:
            return False

        # The board is valid if it does not violate any of the above rules
        return True
