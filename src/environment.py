"""
Grid environment for multi-robot planning under uncertainty.
Models a 2D workspace with obstacles, resource constraints,
and stochastic failure events (hostile attacks, technical glitches).
"""

import numpy as np
from enum import IntEnum
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict


class Action(IntEnum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    STAY  = 4
    RECHARGE = 5


DIRECTIONS = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
    Action.STAY:  ( 0,  0),
}

BATTERY_COST = {
    Action.UP:       1,
    Action.DOWN:     1,
    Action.LEFT:     1,
    Action.RIGHT:    1,
    Action.STAY:     0,
    Action.RECHARGE: 0,
}


class UncertaintyModel:
    """
    Models stochastic uncertainty in robot operations.
    
    Three uncertainty types:
    - Action failure: robot stays put despite issuing a move command
    - Battery drain spike: unexpected extra battery consumed
    - Communication loss: robot loses coordination for one step
    """

    def __init__(self,
                 action_failure_prob: float = 0.10,
                 battery_spike_prob:  float = 0.05,
                 comm_loss_prob:      float = 0.08,
                 seed: int = 42):
        self.action_failure_prob = action_failure_prob
        self.battery_spike_prob  = battery_spike_prob
        self.comm_loss_prob      = comm_loss_prob
        self.rng = np.random.default_rng(seed)

    def apply(self, action: Action, battery: int) -> Tuple[Action, int, bool]:
        """
        Apply uncertainty to intended action.
        Returns (effective_action, extra_battery_cost, comm_lost).
        """
        effective_action = action
        extra_cost = 0
        comm_lost  = False

        # action failure (robot slips, glitch)
        if action != Action.STAY and action != Action.RECHARGE:
            if self.rng.random() < self.action_failure_prob:
                effective_action = Action.STAY

        # battery spike
        if self.rng.random() < self.battery_spike_prob:
            extra_cost = self.rng.integers(1, 3)

        # communication loss
        if self.rng.random() < self.comm_loss_prob:
            comm_lost = True

        return effective_action, extra_cost, comm_lost


class Environment:
    """
    2D grid environment for multi-robot coordination.

    Features:
    - Obstacle-aware grid
    - Multiple charging stations
    - Per-robot battery tracking
    - Stochastic uncertainty events
    - Task locations (goals robots must reach)
    """

    def __init__(self,
                 rows: int = 10,
                 cols: int = 10,
                 n_robots: int = 3,
                 battery_capacity: int = 20,
                 obstacle_ratio: float = 0.15,
                 n_tasks: int = 5,
                 uncertainty: Optional[UncertaintyModel] = None,
                 seed: int = 42):

        self.rows = rows
        self.cols = cols
        self.n_robots = n_robots
        self.battery_capacity = battery_capacity
        self.rng = np.random.default_rng(seed)
        self.uncertainty = uncertainty or UncertaintyModel(seed=seed)

        self.grid = self._generate_grid(obstacle_ratio)
        self.free_cells = [(r, c) for r in range(rows)
                           for c in range(cols) if self.grid[r, c] == 0]

        self.charging_stations = self._place_charging_stations()
        self.task_locations     = self._place_tasks(n_tasks)

        self.robot_positions  = self._init_robots()
        self.robot_batteries  = [battery_capacity] * n_robots
        self.robot_active     = [True] * n_robots   # False = dead battery
        self.robot_comm_lost  = [False] * n_robots

        self.completed_tasks: Set[Tuple[int,int]] = set()
        self.step_count  = 0
        self.history: List[Dict] = []

    # ── Grid generation ──────────────────────────────────────────────────────

    def _generate_grid(self, obstacle_ratio: float) -> np.ndarray:
        while True:
            grid = (self.rng.random((self.rows, self.cols)) < obstacle_ratio).astype(int)
            grid[0, 0] = grid[0, -1] = grid[-1, 0] = grid[-1, -1] = 0
            if self._is_connected(grid):
                return grid

    def _is_connected(self, grid: np.ndarray) -> bool:
        from collections import deque
        start = next(((r, c) for r in range(self.rows)
                      for c in range(self.cols) if grid[r, c] == 0), None)
        if not start:
            return False
        visited, queue = {start}, deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return len(visited) == int(np.sum(grid == 0))

    def _place_charging_stations(self, n: int = 3) -> List[Tuple[int,int]]:
        candidates = [c for c in self.free_cells
                      if c not in [(0,0),(0,self.cols-1),(self.rows-1,0)]]
        indices = self.rng.choice(len(candidates),
                                  size=min(n, len(candidates)), replace=False)
        return [candidates[i] for i in indices]

    def _place_tasks(self, n_tasks: int) -> List[Tuple[int,int]]:
        non_station = [c for c in self.free_cells
                       if c not in self.charging_stations]
        indices = self.rng.choice(len(non_station),
                                  size=min(n_tasks, len(non_station)), replace=False)
        return [non_station[i] for i in indices]

    def _init_robots(self) -> List[Tuple[int,int]]:
        starts = self.free_cells[:self.n_robots]
        return list(starts)

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, actions: List[Action]) -> Dict:
        """
        Execute one timestep for all robots simultaneously.
        Returns a dict with observations, rewards, done flags.
        """
        assert len(actions) == self.n_robots
        rewards    = [0.0] * self.n_robots
        done_flags = [False] * self.n_robots
        events     = []

        for i, action in enumerate(actions):
            if not self.robot_active[i]:
                done_flags[i] = True
                continue

            # apply uncertainty
            eff_action, extra_cost, comm_lost = self.uncertainty.apply(
                action, self.robot_batteries[i])
            self.robot_comm_lost[i] = comm_lost

            if comm_lost:
                events.append(f"Robot {i}: communication lost this step")

            # recharge
            if eff_action == Action.RECHARGE:
                pos = self.robot_positions[i]
                if pos in self.charging_stations:
                    recharge_amt = min(5, self.battery_capacity - self.robot_batteries[i])
                    self.robot_batteries[i] += recharge_amt
                    rewards[i] += 2.0
                    if recharge_amt > 0:
                        events.append(f"Robot {i}: recharged +{recharge_amt} at {pos}")
                else:
                    rewards[i] -= 0.5  # penalty for trying to recharge at wrong spot
                continue

            # movement
            if eff_action != Action.STAY:
                dr, dc = DIRECTIONS[eff_action]
                r, c   = self.robot_positions[i]
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and self.grid[nr, nc] == 0):
                    self.robot_positions[i] = (nr, nc)
                else:
                    rewards[i] -= 0.2   # wall collision penalty
                    if eff_action != action:
                        events.append(f"Robot {i}: action failed (glitch), stayed at ({r},{c})")

            # battery consumption
            cost = BATTERY_COST[eff_action] + extra_cost
            self.robot_batteries[i] = max(0, self.robot_batteries[i] - cost)
            if extra_cost > 0:
                events.append(f"Robot {i}: battery spike -{extra_cost}")

            # dead battery
            if self.robot_batteries[i] == 0:
                self.robot_active[i] = False
                done_flags[i] = True
                rewards[i] -= 10.0
                events.append(f"Robot {i}: DEAD — battery depleted at {self.robot_positions[i]}")
                continue

            # task completion
            pos = self.robot_positions[i]
            if pos in self.task_locations and pos not in self.completed_tasks:
                self.completed_tasks.add(pos)
                rewards[i] += 20.0
                events.append(f"Robot {i}: TASK COMPLETE at {pos} 🎯")

            rewards[i] -= 0.1  # small step penalty (encourages efficiency)

        self.step_count += 1
        obs = self._get_observations()
        info = {
            "step": self.step_count,
            "events": events,
            "tasks_completed": len(self.completed_tasks),
            "tasks_total": len(self.task_locations),
            "batteries": list(self.robot_batteries),
            "positions": list(self.robot_positions),
        }
        self.history.append(info)
        return {"observations": obs, "rewards": rewards,
                "done": done_flags, "info": info}

    def _get_observations(self) -> List[Dict]:
        obs = []
        for i in range(self.n_robots):
            obs.append({
                "position":    self.robot_positions[i],
                "battery":     self.robot_batteries[i],
                "active":      self.robot_active[i],
                "comm_lost":   self.robot_comm_lost[i],
                "nearby_stations": [
                    s for s in self.charging_stations
                    if abs(s[0]-self.robot_positions[i][0]) +
                       abs(s[1]-self.robot_positions[i][1]) <= 3
                ],
                "nearby_tasks": [
                    t for t in self.task_locations
                    if t not in self.completed_tasks and
                       abs(t[0]-self.robot_positions[i][0]) +
                       abs(t[1]-self.robot_positions[i][1]) <= 4
                ],
            })
        return obs

    def reset(self):
        self.robot_positions  = self._init_robots()
        self.robot_batteries  = [self.battery_capacity] * self.n_robots
        self.robot_active     = [True] * self.n_robots
        self.robot_comm_lost  = [False] * self.n_robots
        self.completed_tasks  = set()
        self.step_count       = 0
        self.history          = []

    @property
    def all_done(self) -> bool:
        return (len(self.completed_tasks) == len(self.task_locations) or
                not any(self.robot_active))

    def __repr__(self):
        return (f"Environment({self.rows}×{self.cols}, "
                f"{self.n_robots} robots, "
                f"{len(self.task_locations)} tasks, "
                f"{len(self.charging_stations)} stations)")