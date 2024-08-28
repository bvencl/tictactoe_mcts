import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
from typing import Optional


class Board:
    def __init__(self) -> None:
        self.positions = torch.zeros((3, 3), dtype=torch.int)

    def step(self, action, player):
        self.positions.view(3**2)[action] = player
        return self.positions

    def reset(self) -> None:
        self.positions = torch.zeros((3, 3), dtype=torch.int)
        print("Game reseted")

    def render(self, iteration_time: Optional[float]) -> None:
        print(self.positions)
        cmap = ListedColormap(["white", "blue", "red"])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(self.positions, cmap=cmap, norm=norm)
        if iteration_time is not None:
            ax.text(
                0.5,
                1.05,
                f"Iteration time: {iteration_time:.4f} seconds",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )

        for i in range(3):
            for j in range(3):
                if self.positions[i, j] == 1:
                    ax.text(
                        j,
                        i,
                        "X",
                        ha="center",
                        va="center",
                        fontsize=30,
                        color="black",
                    )
                elif self.positions[i, j] == -1:
                    ax.text(
                        j,
                        i,
                        "O",
                        ha="center",
                        va="center",
                        fontsize=30,
                        color="black",
                    )

        plt.show()

    def turn_table(self) -> None:
        self.positions = -1 * self.positions

    def is_terminal(self) -> Optional[int]:
        row_sum = self.positions.sum(dim=1)
        column_sum = self.positions.sum(dim=0)

        if any(row_sum == 3) or any(column_sum == 3):
            return 1
        if any(row_sum == -3) or any(column_sum == -3):
            return -1

        diagonal1 = (
            self.positions.view(3**2)[0]
            + self.positions.view(3**2)[4]
            + self.positions.view(3**2)[8]
        )
        diagonal2 = (
            self.positions.view(3**2)[2]
            + self.positions.view(3**2)[4]
            + self.positions.view(3**2)[6]
        )

        if diagonal1 == 3 or diagonal2 == 3:
            return 1
        if diagonal1 == -3 or diagonal2 == -3:
            return -1
        if torch.all(self.positions != 0):
            return 0
        return None

    def is_taken(self, position) -> bool:
        if self.positions.view(3**2)[position] != 0:
            return False
        return True

    def free_squares(self) -> list[int]:
        return [
            i for i, square in enumerate(self.positions.view(3**2)) if square == 0
        ] or []
