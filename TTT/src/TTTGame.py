from TTT.src.Player import Player
from TTT.src.RLPlayer import RLPlayer
from TTT.src.TTTBoard import Board
class Game:
    player1 = None;
    player2 = None;
    board = None;

    def __init__(self):

        self.board = Board()

    def playTwoHuman(self):
        '''
        play the game with two human player, terminate once one of the player win.
        :return:
        '''
        self.player1 = Player("X", 1, 1)
        self.player2 = Player("O", 2, -1)
        flip = 1;
        while 1:
            if flip == 1:
                player = self.player1
            else:
                player = self.player2
            inputData = input("Player " + str(player.index) + ", please input row and column that you want to put on the board with format row,col\n")
            row = int(inputData.split(',')[0])
            col = int(inputData.split(',')[1])
            while self.board.set(player,row,col) != True:
                inputData = input("Player " + str(
                player.index) + ", please input row and column that you want to put on the board with format row,col\n")
                row = int(inputData.split(',')[0])
                col = int(inputData.split(',')[1])
            if self.board.isWin(player,row,col):
                self.board.printBoard()
                print("Congradulation! Player " + str(player.index) + ", you have win the game")
                break;
            if self.board.isDraw():
                self.board.printBoard()
                print("The game is draw")
                break;
            self.board.printBoard()


            flip = (flip + 1)%2

    def playRLPlayer(self):
        self.player1 = Player("X", 1, 1)
        self.player2 = RLPlayer("O", 2, -1,"valueFunctionTable.txt", load=True)
        flip = 1;
        while 1:
            if flip == 1:
                inputData = input("Player 1, please input row and column that you want to put on the board with format row,col\n")
                row = int(inputData.split(',')[0])
                col = int(inputData.split(',')[1])
                while self.board.set(self.player1,row,col) != True:
                    inputData = input(
                        "Player 1, please input row and column that you want to put on the board with format row,col\n")
                    row = int(inputData.split(',')[0])
                    col = int(inputData.split(',')[1])
                if self.board.isWin(self.player1,row,col):
                    self.board.printBoard()
                    print("Congradulation! You have win the game")
                    previousBoard = self.board.clone()
                    previousBoard.getBoard()[row][col] = ' '
                    self.player2.updateValueFunctionTable(previousBoard, self.board)
                    self.player2.outputValueFunctionTable("valueFunctionTable.txt")
                    break;

            else:
                move = self.player2.decideMove(self.board)
                self.board.set(self.player2, move[0], move[1])

                if self.board.isWin(self.player2,move[0],move[1]):
                    self.board.printBoard()
                    print("You lost the game")
                    self.player2.outputValueFunctionTable("valueFunctionTable.txt")
                    break;
            if self.board.isDraw():
                self.board.printBoard()
                print("The game is draw")
                self.player2.outputValueFunctionTable("valueFunctionTable.txt")
                break;

            self.board.printBoard()


            flip = (flip + 1)%2


    #def trainRLPlayer(self):



g = Game()
#g.playTwoHuman()
g.playRLPlayer()