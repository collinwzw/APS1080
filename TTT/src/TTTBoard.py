# 1. Create a python class maintains the TTT board. It should have a reset method (to clear the TTT board),
# methods for setting the position of the board for each player, and a method to indicate whether the game has
# been won (returning who has won), or whether the game is at a draw.
import copy


class Board:
    __boardsize = 3
    __borad = None

    __rowCount = None
    __colCount = None
    __forwardDiagonalCount = None
    __backwardDiagonalCount = None
    __count = None;
    def __init__(self):
        #constructor
        self.__borad= [[' ' for col in range(self.__boardsize)] for row in range(self.__boardsize   )]
        self.__rowCount = [0  for row in range(self.__boardsize)]
        self.__colCount = [0  for row in range(self.__boardsize)]
        self.__forwardDiagonalCount = 0
        self.__backwardDiagonalCount = 0
        self.__count = 0

    def printBoard(self):
        '''
        print out the current board
        '''
        for col in range(self.__boardsize):
            for row in range(self.__boardsize):
                print(self.__borad[col][row], end='')
                if (row == self.__boardsize - 1):
                    if col == self.__boardsize - 1:
                        continue
                    else:
                        print('');
                        print("-----")
                else:
                    print('|', end='')
        print("")

    def getBoard(self):
        '''
        method to get 2D array format of board
        :return: 2D array
        '''
        return self.__borad

    def set(self, player, row, col ):
        '''
        This function set the board with given input data
        :param player: Player class to indicate which player is setting the board
        :param row: the row position where play want to set
        :param col: the col position where play want to set
        :return: boolean if palyer is successfully set the board
        '''
        if col >= self.__boardsize or row >= self.__boardsize or self.__borad[row][col] != ' ':
            print("the input position is invalid, please try again")
            return False
        else:
            self.__borad[row][col] = player.playerSymbol
            self.__count = self.__count + 1
            return True

    def resetBoard(self):
        '''
        This method reset the board to completely empty
        :return:
        '''
        self.__borad = [[' ' for col in range(self.__boardsize)] for row in range(self.__boardsize)]
        self.__rowCount = [0  for row in range(self.__boardsize)]
        self.__colCount = [0  for row in range(self.__boardsize)]
        self.__forwardDiagonalCount = 0
        self.__backwardDiagonalCount = 0

    def clone(self):
        return copy.deepcopy(self)

    def isWin(self,player, row,col):
        '''
        check if the one of the player win the game or not with constant time
        :param player: the current player
        :param row: the row position where play want to set
        :param col: the col position where play want to set
        :return: true is one of player win. else False
        '''
        self.__rowCount[row] = self.__rowCount[row] + player.increment
        if self.__rowCount[row] == 3 or self.__rowCount[row] == -3:
            return True
        self.__colCount[col] = self.__colCount[col] + player.increment
        if self.__colCount[col] == 3 or self.__colCount[col] == -3:
            return True
        if col == row:
            self.__forwardDiagonalCount = self.__forwardDiagonalCount + player.increment
            if self.__forwardDiagonalCount == 3 or self.__forwardDiagonalCount == -3:
                return True

        if (row == col and col == 1) or row - col == 2 or col - row == 2 :
            self.__backwardDiagonalCount = self.__backwardDiagonalCount + player.increment
            if self.__backwardDiagonalCount == 3 or self.__backwardDiagonalCount == -3:
                return True
        return False

    def isDraw(self):
        if self.__count  == pow(self.__boardsize,2) :
            return True
        return False

