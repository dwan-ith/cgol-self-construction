"""
Agent definitions for Conway's Game of Life and Self-Construction.
Compliant with Mesa 3.x discrete space FixedAgent patterns.
"""

from typing import Optional, Any
from mesa.discrete_space import FixedAgent


class ConstructionCell(FixedAgent):
    """
    Represents an individual cellular agent on the simulation grid.
    Compatible with Mesa's standard Conway's Game of Life agent design.
    """

    DEAD = 0
    ALIVE = 1

    def __init__(self, model: Any, cell: Any, init_state: int = DEAD):
        super().__init__(model)
        self.cell = cell
        self.state = init_state
        self._next_state: Optional[int] = None
        self.component_type = "empty"  # 'empty', 'glider', 'elbow', 'target', 'debris'

    @property
    def x(self) -> int:
        return self.cell.coordinate[0]

    @property
    def y(self) -> int:
        return self.cell.coordinate[1]

    @property
    def is_alive(self) -> bool:
        return self.state > 0

    @property
    def neighbors(self):
        return self.cell.neighborhood.agents

    def determine_state(self) -> None:
        """Compute next state based on Moore neighborhood for Conway Life."""
        live_neighbors = sum(neighbor.is_alive for neighbor in self.neighbors)
        self._next_state = self.state

        if self.is_alive:
            if live_neighbors < 2 or live_neighbors > 3:
                self._next_state = self.DEAD
        else:
            if live_neighbors == 3:
                self._next_state = self.ALIVE

    def assume_state(self) -> None:
        """Apply computed next state."""
        self.state = self._next_state
