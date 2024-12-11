import random
 
#prompts user to enter if they would like to be x or o 
def select_letter():
  user = ""
  comp = ""
  #ask user to select a letter (X or O), will repeat if input is invalid
  while(user != "x" and user != "o"):
    user = input("Select x or o: ")
    if user == "x":
      comp = "o"
    else:
      comp = "x"
  return user, comp #returns letter user and computer will be using (comp is computer letter and user is user letter)
 
#gives a clear board to play game
def clean_board():
  #empty board for X and O values
  brd = [" " for x in range(10)]
  return brd
 
#puts letter in desired position 
def insert_letter(board,letter,pos):
  board[pos] = letter
 
#for computers move
def computer_move(board,letter):
  computer_letter = letter
  possible_moves = []
  available_corners = []
  available_edges = []
  available_center = []
  position =- 1
 
   #all possible moves
  for i in range(1,len(board)):
    if board[i] ==" ":
      possible_moves.append(i)
 
  #the computer will choose position to win or block a winning move of the user
  for user in ["x","o"]:
    for i in possible_moves:
      board_copy = board[:]
      board_copy[i] = user
      if is_winner(board_copy,user):
        position = i
 
 
   #if computer cannot win or block, then it will choose a random position starting with the corners, the center then the edges
  if position == -1:
    for i in range(len(board)):
      #an empty index on the board
      if board[i] == " ":
        if i in [1,3,7,9]:
          available_corners.append(i)
        if i == 5:
          available_center.append(i)
        if i in [2,4,6,8]:
          available_edges.append(i)
       
      #check corners 
      if len(available_corners) > 0:
        #select a random position in the corners
        position = random.choice(available_corners)
      #checks the availability of the center
      elif len(available_center) > 0:
        #selects the center as the position
        position = available_center[0]
      #checks the availability of the edges
      elif len(available_edges) > 0:
        #selects a random position in the edges
        position = random.choice(available_edges)
  #fill the position selected with the letter the computer is (x or o)
  board[position] = computer_letter
 
#for drawing the board 
def draw_board(board):
  print(board[1] + "|" + board[2] + "|" + board[3] + "     1 2 3")
  print("-+-+-")
  print(board[4] + "|" + board[5] + "|" + board[6] + "     4 5 6")
  print("-+-+-")
  print(board[7] + "|" + board[8] + "|" + board[9] + "     7 8 9")
  return board
 
#checks if a the player or computer is the winner
def is_winner(board,letter):
  return (board[1] == letter and board[2] == letter and board[3] == letter) or \
  (board[4] == letter and board[5] == letter and board[6] == letter) or \
  (board[7] == letter and board[8] == letter and board[9] == letter) or \
  (board[1] == letter and board[4] == letter and board[7] == letter) or \
  (board[2] == letter and board[5] == letter and board[8] == letter) or \
  (board[3] == letter and board[6] == letter and board[9] == letter) or \
  (board[1] == letter and board[5] == letter and board[9] == letter) or \
  (board[3] == letter and board[5] == letter and board[7] == letter)
 
#repeats the game
def repeat_game():
 
  repeat = input("\nWould you like to play again? \nEnter y for yes or n for no: ")
  while repeat != "n" and repeat != "y":
    repeat = input("Invalid input. Please press y for yes and n for no, and make sure it is a lowercase character: ")
  return repeat
 
#for playing the game
def play_game(): 
  count = 0 
  letter, auto_letter= select_letter()
  
  #clears the board
  board = clean_board()
  print()
  board = draw_board(board)
 
  #check if there are empty positions on the board
  for i in range(10): #helps for checking if there is a tie 
    try:
      position = int(input("\nWhat is your move? (1-9): " ))
 
    except: #to avoid error and program terminating 
      position = int(input("Invalid input. Please enter a position using only numbers between 1 and 9: "))
      
    print() 
    
    #checks if user selects a position out of range 
    while position not in range(1,10):
      position = int(input("Invalid input. Please choose another position to place an "+letter+" between 1 and 9: "))
 
    #checks if user selects an occupied position by X or O
    while board[position] != " ":
      position = int(input("Invalid input. Please choose an empty position to place an "+letter+" between 1 and 9: "))
 
    #put the letter in the selected position and then the computer plays, board is drawn after 
    insert_letter(board,letter,position)
    count += 1

    draw_board(board)
    #computer move
    print("\nThe computer will now go... \n")
    computer_move(board,auto_letter)  
    count += 1
    #draws the board
    draw_board(board)
    
    #checks if anyone won
    if is_winner(board,letter):
      print("\nCongratulations! You Won!!\n")
      board=draw_board(board)
      return repeat_game()
    elif is_winner(board,auto_letter):
      print("\nNice try! The computer won.\n")
      board=draw_board(board)
      return repeat_game()
    #checks if its a tie and the board is full 
    if count >= 9:
      print("\nIts a tie!!\n")
      board=draw_board(board)
      return repeat_game()
    
 
#Starts the game
print("Welcome to Tic Tac Toe. This is a one player game against the comptuer to get three of your chosen letters in a row.")
repeat = "y"
while(repeat == "y"):
  repeat = play_game()
