from TTT.src.Player import Player
from TTT.src.TTTBoard import Board
class Game:
    player1 = None;
    player2 = None;
    board = None;

    def __init__(self):
        self.player1 = Player("X", 1, 1)
        self.player2 = Player("O", 2, -1)
        self.board = Board()

    def playTwoHuman(self):
        '''
        play the game with two human player, terminate once one of the player win.
        :return:
        '''
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



g = Game()
g.playTwoHuman()
