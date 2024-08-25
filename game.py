from tictactoe import Board
from mcts import Agent


class Game:

    def __init__(self):
        self.turncount = 0
        self.board = Board()
        self.agent = Agent(self.board)

    def game(self):
        self.agent.Turn(500)

game = Game()
game.game()