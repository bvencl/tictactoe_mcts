from tictactoe import Board
import constants
import numpy as np
from typing import Optional, List
import copy, random, sys
# from mcts import side

side = 0


class Node:
    def __init__(
        self, parent_node: Optional["Node"], board: Board, action: Optional[int], player
    ) -> None:

        self.parent = parent_node
        self.action = action
        self.player = player
        self.visit_count = [0, 0]
        self.score = [0.0, 0.0]
        self.ucb1 = [np.inf, np.inf]
        self.state = copy.deepcopy(board)
        if self.action is not None:
            self.state.step(self.action, self.player)
        self.children = []
        self.legal_actions = self.state.free_squares()
        self.is_terminal_node = self.state.is_terminal()
        if self.player == 1:
            self.side = 0
        else:
            self.side = 1

    def evaluate(self) -> Optional[List[float]]:
        terminal_state = self.state.is_terminal()
        if terminal_state == 1:
            return [constants.WIN_REWARD, constants.LOSS_REWARD]
        if terminal_state == -1:
            return [constants.LOSS_REWARD, constants.WIN_REWARD]
        if terminal_state == 0:
            return [constants.DRAW_REWARD, constants.DRAW_REWARD]
        else:
            return None

    def is_leaf(self):
        if self.children == [] or self.is_terminal_node is not None:
            return True
        return False

    def expand_node(self):
        if self.is_terminal_node is not None:
            return self.evaluate()
        else:
            self.children = [
                Node(
                    parent_node=self,
                    board=self.state,
                    action=action,
                    player=-self.player,
                )
                for action in self.legal_actions
            ]
            self.children_count = len(self.children)

    def best_child(self) -> "Node":
        if self.is_terminal_node is not None:
            return self
        for child in self.children:
            if child is not None:
                child.calculate_ucb1()
        best = self.children[0]
        for child in self.children:
            if child is not None and child.ucb1[side % 2] > best.ucb1[side % 2]:
                best = child
        return best

    def robust_child(self) -> "Node":
        if not self.children:
            raise ValueError("No children to choose from in most_robust_child.")
        best = max(
            self.children,
            key=lambda child: child.visit_count[side % 2],
        )
        return best

    def confident_child(self) -> "Node":
        if not self.children:
            raise ValueError("No children to choose from in confident_child.")
        best = max(
            (child for child in self.children if child.ucb1[side % 2] != np.inf),
            key=lambda child: child.ucb1[side % 2],
        )
        return best

    def calculate_ucb1(self) -> List[float]:
        global side
        if self.visit_count[side % 2] == 0:
            self.ucb1[side % 2] = np.inf
            return self.ucb1
        else:
            if self.parent is None:
                exploration_side = 0
            else:
                exploration_side = (
                    2
                    * constants.UCB_EXPLORATION_CONSTANT
                    * np.sqrt(
                        (2 * np.log(self.parent.visit_count[side % 2]))
                        / self.visit_count[side % 2]
                    )
                )

            exploitation_side = self.score[side % 2] / self.visit_count[side % 2]

            ucb1_player_side = exploitation_side + exploration_side

            self.ucb1[side % 2] = ucb1_player_side
            return self.ucb1
