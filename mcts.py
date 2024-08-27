from tictactoe import Board
from node import Node, side
import constants
import numpy as np
from typing import Optional, List
import copy, random, sys

class Agent:

    def __init__(self, current_state: Board):
        self.turncount = 0
        self.current_global_state = current_state
        self.root = Node(parent_node=None, board=current_state, action=None, player=-1)
        self.root.expand_node()
        self.current_global_node = self.root
        self.current_node = self.root

    def Turn(self, depth):
        global twoplayermode
        global side
        while self.current_global_state.is_terminal() is None:
            if side % 2 == 0 and twoplayermode is True:
                player_step = self.player_turn()
                print("-----------------------------------------------fasz")
            else:
                for _ in range(depth):
                    print(_)
                    while not self.current_node.is_leaf():
                        self.current_node = self.current_node.best_child()
                    if self.current_node.visit_count[side % 2] == 0:
                        copied_node = copy.deepcopy(self.current_node)
                        evaluation = self.rollout(copied_node)
                        del copied_node
                        self.backpropagation(evaluation)
                    else:
                        eval = self.current_node.expand_node()
                        if eval is not None:
                            self.backpropagation(eval)
                        else:
                            best_child = self.current_node.best_child()
                            copied_node = copy.deepcopy(best_child)
                            evaluation = self.rollout(copied_node)
                            del copied_node
                            self.backpropagation(evaluation)
                    self.current_node = self.current_global_node

            self.current_global_node = self.current_global_node.robust_child()
            self.update_global_state()
            self.current_global_state.render()
            side += 1

    def backpropagation(self, result):
        global side
        node = self.current_node
        while node is not None:
            node.visit_count[side % 2] += 1
            node.score[0] += result[0]
            node.score[1] += result[1]
            node = node.parent

    def rollout(self, node: Node):
        current_player = node.player
        while node.state.is_terminal() is None:
            node.legal_actions = node.state.free_squares()
            random_action = random.choice(node.legal_actions)
            node.state.step(random_action, -current_player)
            current_player = -current_player
        return node.evaluate()

    def update_global_state(self):
        self.current_global_state.step(
            self.current_global_node.action, self.current_global_node.player
        )

    def player_turn(self) -> int:
        player_step = input("Te jössz bohóc")
        return int(player_step)


class Game:
    def __init__(self):
        self.board = Board()
        self.agent = Agent(self.board)

    def game(self):
        self.agent.Turn(500)


if len(sys.argv) > 1 and sys.argv[1].lower() == "true":
    twoplayermode = True
else:
    twoplayermode = False

game = Game()
game.game()
