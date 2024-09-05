from tictactoe import Board
import constants
from typing import Optional
import copy, random, math


class Node:
    def __init__(
        self, parent_node: Optional["Node"], board: Board, action: Optional[int], player
    ) -> None:

        self.parent = parent_node
        self.action = action
        self.player = player
        self.visit_count = 0
        self.score = 0.0
        self.ucb1 = 0.0
        self.state = copy.deepcopy(board)
        if self.action is not None:
            self.state.step(self.action, self.player)
        self.children = []
        self.legal_actions = self.state.free_squares()
        self.is_terminal_node = self.state.is_terminal()

    def evaluate(self):
        terminal_state = self.state.is_terminal()
        if terminal_state == self.player:
            return constants.WIN_REWARD
        if terminal_state == -self.player:
            return constants.LOSS_REWARD
        if terminal_state == 0:
            return constants.DRAW_REWARD
        else:
            return None

    def is_leaf(self):
        return self.children == [] or self.is_terminal_node is not None

    def expand_node(self):
        if self.is_terminal_node is not None:
            return self.evaluate()
        else:
            self.children = [
                Node(
                    parent_node=self,
                    board=self.state,
                    action=action,
                    player=self.player,
                )
                for action in self.legal_actions
            ]

    def best_child(self) -> "Node":
        if self.is_terminal_node is not None:
            return self
        for child in self.children:
            if child is not None:
                child.ucb1 = child.calculate_ucb1()
        best = self.children[0]
        for child in self.children:
            if child is not None and child.ucb1 > best.ucb1:
                best = child
        return best

    def robust_child(self) -> "Node":
        if not self.children:
            raise ValueError("No children to choose from in robust_child.")
        best = max(
            self.children,
            key=lambda child: child.visit_count,
        )
        return best

    def confident_child(self) -> "Node":
        if not self.children:
            raise ValueError("No children to choose from in confident_child.")
        best = max(
            (child for child in self.children if child.ucb1 != math.inf),
            key=lambda child: child.ucb1,
        )
        return best

    def calculate_ucb1(self) -> float:
        if self.visit_count == 0:
            return math.inf
        else:
            if self.parent == None:
                exploration = 0
            else:
                exploration = (
                    2
                    * constants.UCB_EXPLORATION_CONSTANT
                    * math.sqrt((math.log(self.parent.visit_count)) / self.visit_count)
                )
                exploitation = self.score / self.visit_count
            return exploitation + exploration


class Agent:
    def __init__(self, current_state: Board):
        self.turncount = 0
        self.current_global_state = current_state
        self.root = Node(parent_node=None, board=current_state, action=None, player=-1)
        self.root.expand_node()
        self.current_node = self.root
        self.turncount = 0
        self.iteration_time = 0

    def Turn(self, depth):
        global is_player_first
        if is_player_first:
            self.agent_action = -1
        else:
            self.agent_action = 1
        while self.current_global_state.is_terminal() is None:
            if (self.turncount % 2) + is_player_first == 1:
                self.update_global_state(self.player_turn())
            else:
                for iter in range(depth):
                    print(iter)
                    while not self.current_node.is_leaf():
                        self.current_node = self.current_node.best_child()
                    if self.current_node.visit_count == 0:
                        copied_node = copy.deepcopy(self.current_node)
                        evaluation = self.rollout(copied_node)
                        self.backpropagation(evaluation)
                        del copied_node
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
                    self.current_node = self.root
                self.chosen_node = self.root.confident_child()
                self.update_global_state(None)

            self.root = Node(
                parent_node=None,
                board=self.current_global_state,
                action=None,
                player=self.agent_action,
            )
            self.root.expand_node()
            self.current_node = self.root
            self.current_global_state.render()
            self.turncount += 1

    def update_global_state(self, action: Optional[int]):
        if action is None:
            self.current_global_state.step(self.chosen_node.action, self.agent_action)
        else:
            self.current_global_state.step(action=action, player=-self.agent_action)

    def player_turn(self):
        return int(input("Te jössz bohóc!\n"))

    def backpropagation(self, result):
        node = self.current_node
        while node is not None:
            node.visit_count += 1
            node.score += result
            node = node.parent

    def rollout(self, node: Node):
        current_player = node.player
        while node.state.is_terminal() is None:
            node.legal_actions = node.state.free_squares()
            random_action = random.choice(node.legal_actions)
            node.state.step(random_action, -current_player)
            current_player = -current_player
        return node.evaluate()


class Game:
    def __init__(self):
        self.board = Board()
        self.agent = Agent(self.board)

    def game(self):
        self.agent.Turn(1000)


is_player_first = True
game = Game()
game.game()
