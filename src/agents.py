"""
Agent policies for multi-robot coordination under uncertainty.

Three agents implemented:
1. RandomAgent       - baseline, moves randomly
2. GreedyAgent       - goes to nearest task, recharges when low battery
3. CoordinatedAgent  - task allocation + battery-aware planning (best policy)
"""

import math
import random
from collections import deque
from typing import List, Tuple, Dict, Optional
from environment import Action, Environment


def bfs_path(grid, start: Tuple, goal: Tuple) -> List[Action]:
    """BFS shortest path from start to goal. Returns list of actions."""
    rows, cols = grid.shape
    if start == goal:
        return []

    visited = {start: None}
    queue = deque([(start, [])])

    while queue:
        (r, c), path = queue.popleft()
        for action, (dr, dc) in [
            (Action.UP,    (-1, 0)),
            (Action.DOWN,  ( 1, 0)),
            (Action.LEFT,  ( 0,-1)),
            (Action.RIGHT, ( 0, 1)),
        ]:
            nr, nc = r+dr, c+dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr, nc] == 0 and (nr, nc) not in visited):
                new_path = path + [action]
                if (nr, nc) == goal:
                    return new_path
                visited[(nr, nc)] = True
                queue.append(((nr, nc), new_path))
    return []


def manhattan(a: Tuple, b: Tuple) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


# ─────────────────────────────────────────────────────────────────────────────

class RandomAgent:
    """Baseline: picks a random valid action each step."""

    def __init__(self, robot_id: int):
        self.robot_id = robot_id

    def act(self, obs: Dict, env: Environment) -> Action:
        if not obs["active"]:
            return Action.STAY
        return random.choice(list(Action))

    def __repr__(self):
        return f"RandomAgent(id={self.robot_id})"


# ─────────────────────────────────────────────────────────────────────────────

class GreedyAgent:
    """
    Greedy single-robot policy:
    - If battery < threshold → go to nearest charging station
    - Else → go to nearest uncompleted task
    - Uses BFS for shortest path navigation
    """

    def __init__(self, robot_id: int, battery_threshold: int = 6):
        self.robot_id  = robot_id
        self.threshold = battery_threshold
        self._path: List[Action] = []
        self._target: Optional[Tuple] = None

    def act(self, obs: Dict, env: Environment) -> Action:
        if not obs["active"]:
            return Action.STAY

        pos     = obs["position"]
        battery = obs["battery"]

        # recharge if at station and low
        if pos in env.charging_stations and battery < self.threshold:
            return Action.RECHARGE

        # replan if target reached or path exhausted
        if not self._path:
            self._target = self._choose_target(obs, env)
            if self._target is None:
                return Action.STAY
            self._path = bfs_path(env.grid, pos, self._target)

        if self._path:
            return self._path.pop(0)
        return Action.STAY

    def _choose_target(self, obs: Dict, env: Environment) -> Optional[Tuple]:
        pos     = obs["position"]
        battery = obs["battery"]

        if battery < self.threshold and env.charging_stations:
            return min(env.charging_stations, key=lambda s: manhattan(pos, s))

        remaining = [t for t in env.task_locations
                     if t not in env.completed_tasks]
        if remaining:
            return min(remaining, key=lambda t: manhattan(pos, t))

        if battery < self.threshold and env.charging_stations:
            return min(env.charging_stations, key=lambda s: manhattan(pos, s))

        return None

    def reset(self):
        self._path   = []
        self._target = None

    def __repr__(self):
        return f"GreedyAgent(id={self.robot_id}, threshold={self.threshold})"


# ─────────────────────────────────────────────────────────────────────────────

class CoordinatedAgent:
    """
    Coordinated multi-robot policy with task allocation.

    Key ideas:
    1. Task allocation: robots bid on tasks based on distance + battery cost.
       Each task is claimed by exactly one robot to avoid redundant travel.
    2. Battery-aware planning: maintains a safety buffer. If remaining battery
       < (dist_to_task + dist_task_to_station + buffer), recharge first.
    3. Uncertainty handling: replans every N steps to adapt to stochastic events.
    """

    SAFETY_BUFFER = 4
    REPLAN_EVERY  = 3

    def __init__(self, robot_id: int, n_robots: int):
        self.robot_id = robot_id
        self.n_robots = n_robots
        self._path:     List[Action] = []
        self._target:   Optional[Tuple] = None
        self._mode:     str = "task"   # "task" | "recharge"
        self._step:     int = 0

    def act(self, obs: Dict, env: Environment,
            claimed_tasks: Dict[Tuple, int]) -> Action:
        """
        claimed_tasks: {task_loc: robot_id} — shared across all coordinated agents.
        """
        if not obs["active"]:
            return Action.STAY

        pos     = obs["position"]
        battery = obs["battery"]
        self._step += 1

        # recharge if at station and needed
        if pos in env.charging_stations and battery < env.battery_capacity * 0.5:
            return Action.RECHARGE

        # replan periodically or when path is done
        if not self._path or self._step % self.REPLAN_EVERY == 0:
            self._replan(obs, env, claimed_tasks)

        if self._path:
            return self._path.pop(0)
        return Action.STAY

    def _replan(self, obs: Dict, env: Environment,
                claimed_tasks: Dict[Tuple, int]):
        pos     = obs["position"]
        battery = obs["battery"]

        # check if we need to recharge first
        nearest_station = min(env.charging_stations,
                              key=lambda s: manhattan(pos, s)) \
                          if env.charging_stations else None

        dist_to_station = manhattan(pos, nearest_station) if nearest_station else 999

        # find best unclaimed task for this robot
        remaining = [t for t in env.task_locations
                     if t not in env.completed_tasks
                     and (t not in claimed_tasks or claimed_tasks[t] == self.robot_id)]

        if not remaining:
            # no tasks left, just recharge
            if nearest_station:
                self._path   = bfs_path(env.grid, pos, nearest_station)
                self._target = nearest_station
            return

        # bid: score = distance + battery penalty
        def score(task):
            d = manhattan(pos, task)
            nearest_s_from_task = min(
                (manhattan(task, s) for s in env.charging_stations), default=0)
            total_cost = d + nearest_s_from_task + self.SAFETY_BUFFER
            if total_cost > battery:
                return float('inf')   # can't safely do this task
            return d

        best_task = min(remaining, key=score)

        if score(best_task) == float('inf'):
            # can't safely reach any task → recharge
            self._mode   = "recharge"
            self._target = nearest_station
            self._path   = bfs_path(env.grid, pos, nearest_station) \
                           if nearest_station else []
        else:
            self._mode = "task"
            claimed_tasks[best_task] = self.robot_id
            self._target = best_task
            self._path   = bfs_path(env.grid, pos, best_task)

    def reset(self):
        self._path   = []
        self._target = None
        self._mode   = "task"
        self._step   = 0

    def __repr__(self):
        return f"CoordinatedAgent(id={self.robot_id})"