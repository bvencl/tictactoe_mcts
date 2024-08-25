from tictactoe import Board
import constants
import numpy as np
from typing import Optional, List
import copy, random
from graphviz import Digraph


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
            print("node was expanded")
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
                child.update_ucb1()
        best = self.children[0]
        for child in self.children:
            if child is not None and child.ucb1[self.side] > best.ucb1[self.side]:
                best = child
        return best

    def robust_child(self) -> "Node":
        if not self.children:
            raise ValueError("No children to choose from in most_robust_child.")
        best = max(
            self.children,
            key=lambda child: child.visit_count[self.side],
        )
        return best

    def confident_child(self) -> "Node":
        if not self.children:
            raise ValueError("No children to choose from in confident_child.")
        best = max(
            (child for child in self.children if child.ucb1[self.side] != np.inf),
            key=lambda child: child.ucb1[self.side],
        )
        return best

    def update_ucb1(self) -> List[float]:
        self.ucb1 = self.calculate_ucb1()
        return self.ucb1

    def calculate_ucb1(self) -> List[float]:
        if self.visit_count[0] == 0:
            self.ucb1[0] = np.inf
            was_first = True
        elif self.visit_count[1] == 0:
            self.ucb1[1] = np.inf
            was_first = True
        if was_first:
            return self.ucb1
        else:
            if self.parent is None:
                exploration = 0
            else:
                exploration = (
                    2
                    * constants.UCB_EXPLORATION_CONSTANT
                    * np.sqrt(
                        (2 * np.log(self.parent.visit_count[self.side]))
                        / self.visit_count[self.side]
                    )
                )

            exploitation_player1 = self.score[0] / self.visit_count[0]
            exploitation_player2 = self.score[1] / self.visit_count[0]

            ucb1_player1 = exploitation_player1 + exploration
            ucb1_player2 = exploitation_player2 + exploration

            self.ucb1 = [ucb1_player1, ucb1_player2]
            return self.ucb1

    def to_graphviz(self, dot=None, highlight_node=None):
        if dot is None:
            dot = Digraph()
            dot.node(
                name=str(id(self)),
                label=f"Action: {self.action}\nVisits: {self.visit_count}\nScore: {self.score:.2f}\nUCB1: {self.ucb1:.2f}",
            )

        for child in self.children:
            if child is not None:
                if highlight_node is not None and child == highlight_node:
                    dot.node(
                        name=str(id(child)),
                        label=f"Action: {child.action}\nVisits: {child.visit_count}\nScore: {child.score:.2f}\nUCB1: {child.ucb1:.2f}",
                        color="red",
                        style="filled",
                        fillcolor="red",
                    )
                else:
                    dot.node(
                        name=str(id(child)),
                        label=f"Action: {child.action}\nVisits: {child.visit_count}\nScore: {child.score:.2f}\nUCB1: {child.ucb1:.2f}",
                    )
                dot.edge(str(id(self)), str(id(child)))
                child.to_graphviz(dot, highlight_node)

        return dot


class Agent:
    def __init__(self, current_state: Board):
        self.turncount = 0
        self.current_global_state = current_state
        self.root = Node(parent_node=None, board=current_state, action=None, player=-1)
        self.root.expand_node()
        self.current_global_node = self.root
        self.current_node = self.root

    def Turn(self, depth):
        while self.current_global_state.is_terminal() is None:
            for _ in range(depth):
                print(_)
                while not self.current_node.is_leaf():
                    self.current_node = self.current_node.best_child()
                if self.current_node.visit_count[self.current_node.side] == 0:
                    copied_node = copy.deepcopy(self.current_node)
                    evaluation = self.rollout(copied_node)
                    del copied_node
                else:
                    self.current_node.expand_node()
                    best_child = self.current_node.best_child()
                    copied_node = copy.deepcopy(best_child)
                    evaluation = self.rollout(copied_node)
                    del copied_node

                self.backpropagation(evaluation)
                self.current_node = self.current_global_node

            self.current_global_node = self.current_global_node.robust_child()
            self.update_global_state()
            global_state_to_render = copy.deepcopy(self.current_global_state)
            if self.turncount % 2 == 1:
                global_state_to_render.positions *= -1
            global_state_to_render.render()
            del global_state_to_render
            self.current_global_state.turn_table()
            self.turncount += 1

    def backpropagation(self, result):
        node = self.current_node
        while node is not None:
            node.visit_count[node.side] += 1
            node.score[0] += result[0]
            node.score[1] += result[1]
            node = node.parent

    def rollout(self, node: Node):
        current_player = node.player
        print("Rollout was done")
        while node.state.is_terminal() is None:
            node.legal_actions = node.state.free_squares()
            random_action = random.choice(node.legal_actions)
            node.state.step(random_action, -current_player)
            current_player = -current_player
        return node.evaluate()

    def update_global_state(self):
        self.current_global_state.step(self.current_global_node.action, 1)

    def graphviz_render(self, filename="mcts_tree", highlight_node=None):
        dot = self.root.to_graphviz(highlight_node=highlight_node)
        dot.render(filename, format="svg")


class Game:
    def __init__(self):
        self.board = Board()
        self.agent = Agent(self.board)

    def game(self):
        self.agent.Turn(250)


if True:
    game = Game()
    game.game()
