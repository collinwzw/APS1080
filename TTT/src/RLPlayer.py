from TTT.src.Player import Player
from TTT.src.TTTBoard import Board
import copy

class RLPlayer(Player):
    valueFunctionTable = {}

    def __init__(self, symbol, index, increment, filename, load=True):
        self.playerSymbol = symbol
        self.index = index;
        self.increment = increment
        if load == True:
            self.loadValueFunctionTable(filename)
        else:
            self.__initValueFunctionTable()
            self.outputValueFunctionTable(filename)

    def __initValueFunctionTable(self):
        '''
        initiate the value function table by using actual board and two fake player
        :return:
        '''
        board = Board()
        player1 = Player("X", 1, 1)
        player2 = Player("O", 2, -1)
        self.__getState(board,player1,player2,1)

    def __getAvailableMove(self, board):
        '''
        read the board and obtain all the possible move
        :param board: the Board class
        :return: the list of list of integer that contains all the possible moves
        '''
        availableMove = []
        for row in range(3):
            for col in range(3):
                if board.getBoard()[row][col] == ' ':
                    availableMove.append([row,col])
        return availableMove

    def __getBoardInfo(self, board):
        '''
        return the string format of board for storing as key in dictionary
        :param board:
        :return: String format of the board
        '''
        return board.getBoard().__str__()

    def __getState(self, board, player1, player2, flip):
        '''
        This method using back tracking to compute all the possible states for the value function table and record down the value
        If opponent win, value is 0
        If we win, value is 1
        All other situations including the intermidiate state get value 0.5
        :param board:
        :param player1:
        :param player2:
        :param flip:
        :return:
        '''
        availableMove = self.__getAvailableMove(board)
        if flip == 1:
            currentPlayer = player1
        else:
            currentPlayer = player2

        for move in availableMove:
            currentBoard = copy.deepcopy(board)

            currentBoard.set(currentPlayer, move[0], move[1])

            if self.__getBoardInfo(currentBoard) not in self.valueFunctionTable.keys():
                if currentBoard.isWin(currentPlayer, move[0], move[1]):
                    if flip == 1:
                        # opponent win
                        self.valueFunctionTable[self.__getBoardInfo(currentBoard)] = 0
                        return
                    else:
                        # we win
                        self.valueFunctionTable[self.__getBoardInfo(currentBoard)] = 1
                        return
                self.valueFunctionTable[self.__getBoardInfo(currentBoard)] = 0.5

                if currentBoard.isDraw() == True:
                    return

                self.__getState(currentBoard,player1,player2,(flip + 1)%2) #recurrension call

    def outputValueFunctionTable(self, filename):
        # output value function table to file
        f = open(filename, "w")
        for key in self.valueFunctionTable.keys():
            f.write(key + ':' + str(self.valueFunctionTable.get(key)) + '\n')
        f.close()

    def loadValueFunctionTable(self, filename):
        '''
        load the value function table from file
        :return:
        '''
        f = open(filename, "r")
        for line in f.readlines():
            split = line.split(':')
            self.valueFunctionTable[split[0]] = float(split[1].split("\n")[0])
        f.close()

    def test(self):
        print("success")


rlplayer = RLPlayer("O", 2, -1,"valueFunctionTable.txt", load=True)
rlplayer.test()