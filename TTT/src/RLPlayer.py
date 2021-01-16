from TTT.src.Player import Player
from TTT.src.TTTBoard import Board


class RLPlayer(Player):
    valueFunctionTable = {}
    step = 0.1
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
            currentBoard = board.clone()

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

    def updateValueFunctionTable(self, previousBoard, currentBoard):
        '''
        update the value function table
        :param previousBoard:
        :param currentBoard:
        :return:
        '''
        self.valueFunctionTable[self.__getBoardInfo(previousBoard)] = self.valueFunctionTable[self.__getBoardInfo(previousBoard)] + \
                                                              self.step * (self.valueFunctionTable.get(
            self.__getBoardInfo(currentBoard)) - self.valueFunctionTable.get(self.__getBoardInfo(previousBoard)))

    def decideMove(self, board):
        '''
        This method takes the board and try to make the move that has max possibility to win the game
        Also, it updates the value function table with pre defined step
        :param board:
        :return:
        '''
        availableMove = self.__getAvailableMove(board)
        max = -1
        move = None;
        for currentMove in availableMove:
            currentBoard = board.clone()
            currentBoard.set(self, currentMove[0], currentMove[1])
            if max < self.valueFunctionTable.get(self.__getBoardInfo(currentBoard)):
                max = self.valueFunctionTable.get(self.__getBoardInfo(currentBoard))
                move = currentMove
        self.updateValueFunctionTable(board, currentBoard)
        return move

    def test(self):
        print("success")


