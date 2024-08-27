from tictactoe import Board
import constants
from typing import Optional, List
import copy, random, math, time
import graphviz

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
        self.ucb1 = [math.inf, math.inf]
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
        self.was_selected = False

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
            (child for child in self.children if child.ucb1[side % 2] != math.inf),
            key=lambda child: child.ucb1[side % 2],
        )
        return best

    def calculate_ucb1(self) -> List[float]:
        global side
        if self.visit_count[side % 2] == 0:
            self.ucb1[side % 2] = math.inf
            return self.ucb1
        else:
            if self.parent is None:
                exploration_side = 0
            else:
                exploration_side = (
                    2
                    * constants.UCB_EXPLORATION_CONSTANT
                    * math.sqrt(
                        (2 * math.log(self.parent.visit_count[side % 2]))
                        / self.visit_count[side % 2]
                    )
                )

            exploitation_side = self.score[side % 2] / self.visit_count[side % 2]

            ucb1_player_side = exploitation_side + exploration_side

            self.ucb1[side % 2] = ucb1_player_side
            return self.ucb1


class Agent:

    def __init__(self, current_state: Board):
        self.current_global_state = current_state
        self.root = Node(parent_node=None, board=current_state, action=None, player=-1)
        self.root.expand_node()
        self.current_global_node = self.root
        self.current_node = self.root
        self.tree = {}

    def Turn(self, depth):
        global side
        while self.current_global_state.is_terminal() is None:
            start_time = time.time()
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

            end_time = time.time()
            iteration_time = end_time - start_time
            self.current_global_node.was_selected = True
            self.current_global_node = self.current_global_node.robust_child()
            self.update_tree(self.current_global_node)
            self.update_global_state()
            self.current_global_state.render(iteration_time)
            side += 1

    def backpropagation(self, result):
        global side
        node = self.current_node
        while node is not None:
            node.visit_count[side % 2] += 1
            # node.visit_count[(side + 1) % 2] += 0.2
            node.score[0] += result[0]
            node.score[1] += result[1]
            self.update_tree(node)
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

    def update_tree(self, node):
        if node not in self.tree:
            self.tree[node] = {
                "visit_count": [0, 0],
                "score": [0, 0],
                "action": node.action,
                "selected": node.was_selected,
                "ucb1": node.ucb1,
            }
        self.tree[node]["visit_count"] = node.visit_count
        self.tree[node]["score"] = node.score
        self.tree[node]["selected"] = node.was_selected
        self.tree[node]["ucb1"] = node.ucb1

    def draw_tree(self):
        dot = graphviz.Digraph(comment="MCTS Tree")
        for node, data in self.tree.items():
            node_id = str(id(node))
            action = data["action"]
            label = f"Action: {action}\nVisits: {data['visit_count']}\n, Score: {data['score']}\n, UCB1: {data['ucb1']}"
            if data["selected"]:
                dot.node(node_id, label, style="filled", fillcolor="red")
            else:
                dot.node(node_id, label)
            if node.parent:
                parent_id = str(id(node.parent))
                dot.edge(parent_id, node_id)
        dot.render("mcts_tree", format="svg")


class Game:
    def __init__(self):
        self.board = Board()
        self.agent = Agent(self.board)

    def game(self):
        self.agent.Turn(500)


if True:
    game = Game()
    game.game()
    game.agent.draw_tree()
