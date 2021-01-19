from TTT.src.Player import Player
from TTT.src.RLPlayer import RLPlayer
from TTT.src.TTTBoard import Board
from random import randrange

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

    def playTwoRLPlayer(self):
        #self.player1 = RLPlayer("X", 1, 1,"valueFunctionTable1.txt",load=False, opposite=True)
        self.player1 = RLPlayer("X", 1, 1, "valueFunctionTable1.txt", load=True)
        self.player2 = RLPlayer("O", 2, -1,"valueFunctionTable.txt", load=True)
        flip = 1;
        while 1:
            if flip == 1:
                move = self.player1.decideMove(self.board)
                self.board.set(self.player1, move[0], move[1])
                if self.board.isWin(self.player1, move[0], move[1]):
                    self.board.printBoard()
                    previousBoard = self.board.clone()
                    previousBoard.getBoard()[move[0]][move[1]] = ' '
                    self.player2.updateValueFunctionTable(previousBoard, self.board)
                    self.player2.outputValueFunctionTable("valueFunctionTable.txt")
                    self.player1.outputValueFunctionTable("valueFunctionTable1.txt")
                    print("player 1 win")
                    break;

            else:
                move = self.player2.decideMove(self.board)
                self.board.set(self.player2, move[0], move[1])

                if self.board.isWin(self.player2,move[0],move[1]):
                    self.board.printBoard()
                    print("player 2 win")
                    previousBoard = self.board.clone()
                    previousBoard.getBoard()[move[0]][move[1]] = ' '
                    self.player1.updateValueFunctionTable(previousBoard, self.board)
                    self.player1.outputValueFunctionTable("valueFunctionTable1.txt")
                    self.player2.outputValueFunctionTable("valueFunctionTable.txt")
                    break;

            if self.board.isDraw():
                self.board.printBoard()
                print("The game is draw")
                self.player2.outputValueFunctionTable("valueFunctionTable.txt")
                break;

            self.board.printBoard()
            flip = (flip + 1)%2

    def trainRLPlauer(self, numberOfGames):
        self.player1 = Player("X", 1, 1)
        self.player2 = RLPlayer("O", 2, -1,"valueFunctionTable.txt", load=True)
        flip = 1;
        draw = 0
        randomWin = 0
        rLWin = 0

        while 1:
            if flip == 1:
                availableMove = []
                for row in range(3):
                    for col in range(3):
                        if self.board.getBoard()[row][col] == ' ':
                            availableMove.append([row, col])
                move = availableMove[randrange(len(availableMove))]
                self.board.set(self.player1, move[0], move[1])
                if self.board.isWin(self.player1,move[0], move[1]):
                    #self.board.printBoard()
                    print("random player win the game")
                    previousBoard = self.board.clone()
                    previousBoard.getBoard()[move[0]][move[1]] = ' '
                    self.player2.updateValueFunctionTable(previousBoard, self.board)
                    self.player2.outputValueFunctionTable("valueFunctionTable.txt")
                    self.board.resetBoard()
                    randomWin = randomWin + 1
                    print("total game played = " + str(draw + randomWin + rLWin) + "， random player wins " + str(randomWin) + ", computer wins " +str(rLWin) + ", and draw " + str(draw) + "games" )
                    if draw + randomWin + rLWin >= numberOfGames: break;
                    flip = 0;
            else:
                move = self.player2.decideMove(self.board)
                self.board.set(self.player2, move[0], move[1])

                if self.board.isWin(self.player2,move[0],move[1]):
                    #self.board.printBoard()
                    print("Computer win the game")
                    self.player2.outputValueFunctionTable("valueFunctionTable.txt")
                    self.board.resetBoard()
                    rLWin = rLWin + 1
                    print("total game played = " + str(draw + randomWin + rLWin) + "， random player wins " + str(randomWin) + ", computer wins " +str(rLWin) + ", and draw " + str(draw) + "games" )
                    if draw + randomWin + rLWin  >= numberOfGames: break;
                    flip = 0;
            if self.board.isDraw():
                #self.board.printBoard()
                print("The game is draw")

                self.player2.outputValueFunctionTable("valueFunctionTable.txt")
                self.board.resetBoard()
                flip = 0;
                draw = draw + 1
                print("total game played = " + str(draw + randomWin + rLWin) + "， random player wins " + str(randomWin) + ", computer wins " +str(rLWin) + ", and draw " + str(draw) + "games" )
                if draw + randomWin + rLWin  >= numberOfGames: break;
            #self.board.printBoard()


            flip = (flip + 1)%2


g = Game()
#g.playTwoHuman()
#g.playRLPlayer()
#g.playTwoRLPlayer()
g.trainRLPlauer(50000)