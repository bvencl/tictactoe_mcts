from tictactoe import Board
import constants
import numpy as np
from typing import Optional
import copy, random
from graphviz import Digraph


class Node:
    def __init__(
        self, parent_node: Optional["Node"], board: Board, action: Optional[int], player
    ) -> None:

        self.parent = parent_node
        self.my_action = action
        self.player = player
        self.state: "Board" = copy.deepcopy(board)
        if self.my_action is not None:
            self.state.positions = self.state.step(self.my_action, player=self.player)

        self.visit_count = 0
        self.score = 0.0
        self.all_actions = self.state.free_squares()
        self.ucb1 = self.calculate_ucb1()
        self.children = []

        self.action_count = len(self.all_actions)
        self.children_count = len(self.children)

    def print_counts(self):
        print(self.children_count, self.visit_count, self.score, self.ucb1)

    def expand_node(self) -> None:
        if not self.all_actions:
            raise ValueError("No children to choose from in best_child.")
        else:
            self.children = [
                Node(
                    parent_node=self,
                    board=self.state,
                    action=action,
                    player=-self.player,
                )
                for action in self.all_actions
            ]
            self.children_count = len(self.children)

    def is_leaf(self):
        if self.children == [] or self.state.is_terminal() is not None:
            return True
        return False

    def best_child(self) -> "Node":
        if not self.children:
            self.to_graphviz().render("mcts_tree_error", format="png")
            raise ValueError("No children to choose from in best_child.")
        for child in self.children:
            if child is not None:
                child.update_ucb1()
        
        best = max(
            (child for child in self.children if child.ucb1 != np.inf),
            key=lambda child: child.ucb1,
            default=None,
        )
        if best is None:
            self.to_graphviz().render("mcts_tree_error", format="png")
            raise ValueError("No valid children with finite ucb1 value.")
        return best

    def most_robust_child(self) -> "Node":
        if not self.children:
            self.to_graphviz().render("mcts_tree_error", format="png")
            raise ValueError("No children to choose from in most_robust_child.")
        best = max(self.children, key=lambda child: child.visit_count)
        return best

    def update_ucb1(self) -> float:
        self.ucb1 = self.calculate_ucb1()
        return self.ucb1

    def calculate_ucb1(self) -> float:
        if self.visit_count == 0:
            return np.inf
        else:
            if self.parent == None:
                exploration = 0
            else:
                exploration = (
                    2
                    * constants.UCB_EXPLORATION_CONSTANT
                    * np.sqrt((2 * np.log(self.parent.visit_count)) / self.visit_count)
                )
                exploitation = self.score / self.visit_count
            return exploitation + exploration

    def to_graphviz(self, dot=None):
        if dot is None:
            dot = Digraph()
            dot.node(
                name=str(id(self)),
                label=f"Action: {self.my_action}\nVisits: {self.visit_count}\nScore: {self.score:.2f}\nUCB1: {self.ucb1:.2f}",
            )

        for child in self.children:
            if child is not None:
                dot.node(
                    name=str(id(child)),
                    label=f"Action: {child.my_action}\nVisits: {child.visit_count}\nScore: {child.score:.2f}\nUCB1: {child.ucb1:.2f}",
                )
                dot.edge(str(id(self)), str(id(child)))
                child.to_graphviz(dot)

        return dot


class Agent:
    def __init__(self, current_state: Board):
        self.current_global_state = current_state
        self.root = Node(parent_node=None, board=current_state, action=None, player=-1)
        self.root.expand_node()
        self.current_global_node = self.root
        self.current_node = self.root

    def Turn(self, depth):
        turncount = 0
        while self.current_global_state.is_terminal() is None:
            if turncount == 1:
                print("fasz")
            for _ in range(depth):
                print(_)
                while not self.current_node.is_leaf():
                    self.current_node = self.current_node.best_child()
                if self.current_node.visit_count == 0:
                    evaluation = self.rollout(copy.deepcopy(self.current_node))
                else:
                    self.current_node.expand_node()
                    best_child = self.current_node.best_child()
                    evaluation = self.rollout(copy.deepcopy(best_child))
                self.backpropagation(evaluation)
                self.current_node = self.current_global_node

            self.current_global_node = self.current_global_node.most_robust_child()
            self.update_global_state()
            golbal_state_to_render = copy.deepcopy(self.current_global_state)
            if turncount % 2 == 1:
                golbal_state_to_render.positions *= -1
            # golbal_state_to_render.render()
            self.current_global_state.turn_table()
            turncount += 1

    def update_global_state(self):
        self.current_global_state.step(self.current_global_node.my_action, 1)

    def backpropagation(self, result):
        node = self.current_node
        while node is not None:
            node.visit_count += 1
            node.score += result
            node = node.parent

    def rollout(self, node: Node):
        current_state = node.state
        current_player = node.player
        while current_state.is_terminal() is None:
            node.all_actions = current_state.free_squares()
            random_action = random.choice(node.all_actions)
            current_state.step(random_action, -current_player)
            current_player = -current_player
        return self.evaluate(current_state, node.player)

    def evaluate(self, current_state: Board, player):
        terminal_state = current_state.is_terminal()
        if terminal_state == player:
            return constants.WIN_REWARD
        if terminal_state == -player:
            return constants.LOSS_REWARD
        if terminal_state == 0:
            return constants.DRAW_REWARD
        else:
            return None


class Game:
    def __init__(self):
        self.board = Board()
        self.agent = Agent(self.board)

    def game(self):
        self.agent.Turn(500)


game = Game()
game.game()
game.agent.current_global_state.render()
dot = game.agent.root.to_graphviz()
dot.render("mcts_tree", format="png")
