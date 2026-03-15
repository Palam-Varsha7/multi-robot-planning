"""
Simulation runner and visualization for multi-robot planning.
Compares Random, Greedy, and Coordinated agent policies.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Dict, Tuple
from environment import Environment, UncertaintyModel, Action
from agents import RandomAgent, GreedyAgent, CoordinatedAgent


ROBOT_COLORS = ["#1565C0", "#B71C1C", "#2E7D32", "#6A1B9A", "#E65100"]

LEGEND_ITEMS = {
    "obstacle":  ("#2c2c2c", "Obstacle"),
    "free":      ("#f0f4f8", "Free cell"),
    "station":   ("#FFD600", "Charging station"),
    "task_open": ("#EF9A9A", "Pending task"),
    "task_done": ("#A5D6A7", "Completed task"),
}


def run_episode(env: Environment, policy: str = "coordinated",
                max_steps: int = 200, verbose: bool = False) -> Dict:
    """
    Run one full episode with the given policy.
    Returns episode statistics.
    """
    env.reset()

    if policy == "random":
        agents = [RandomAgent(i) for i in range(env.n_robots)]
    elif policy == "greedy":
        agents = [GreedyAgent(i) for i in range(env.n_robots)]
    elif policy == "coordinated":
        agents = [CoordinatedAgent(i, env.n_robots) for i in range(env.n_robots)]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    claimed_tasks = {}
    total_rewards = [0.0] * env.n_robots
    battery_log   = [[env.battery_capacity] for _ in range(env.n_robots)]
    position_log  = [[(env.robot_positions[i])] for i in range(env.n_robots)]

    for step in range(max_steps):
        if env.all_done:
            break

        actions = []
        for i, agent in enumerate(agents):
            obs = env._get_observations()[i]
            if policy == "coordinated":
                a = agent.act(obs, env, claimed_tasks)
            else:
                a = agent.act(obs, env)
            actions.append(a)

        result = env.step(actions)

        for i in range(env.n_robots):
            total_rewards[i] += result["rewards"][i]
            battery_log[i].append(env.robot_batteries[i])
            position_log[i].append(env.robot_positions[i])

        if verbose and result["info"]["events"]:
            print(f"  Step {step+1}: {', '.join(result['info']['events'])}")

    return {
        "policy":           policy,
        "steps":            env.step_count,
        "tasks_completed":  len(env.completed_tasks),
        "tasks_total":      len(env.task_locations),
        "completion_rate":  len(env.completed_tasks) / len(env.task_locations),
        "total_reward":     sum(total_rewards),
        "robots_alive":     sum(env.robot_active),
        "battery_log":      battery_log,
        "position_log":     position_log,
        "final_batteries":  list(env.robot_batteries),
    }


def compare_policies(env: Environment, n_trials: int = 10) -> Dict:
    """Run multiple trials for each policy and average results."""
    policies = ["random", "greedy", "coordinated"]
    stats = {p: {"completion_rates": [], "steps": [], "rewards": [], "alive": []}
             for p in policies}

    for p in policies:
        for trial in range(n_trials):
            uncertainty = UncertaintyModel(seed=trial * 7)
            env.uncertainty = uncertainty
            result = run_episode(env, policy=p, max_steps=200)
            stats[p]["completion_rates"].append(result["completion_rate"])
            stats[p]["steps"].append(result["steps"])
            stats[p]["rewards"].append(result["total_reward"])
            stats[p]["alive"].append(result["robots_alive"])

    summary = {}
    for p in policies:
        summary[p] = {
            "avg_completion": np.mean(stats[p]["completion_rates"]),
            "avg_steps":      np.mean(stats[p]["steps"]),
            "avg_reward":     np.mean(stats[p]["rewards"]),
            "avg_alive":      np.mean(stats[p]["alive"]),
            "std_completion": np.std(stats[p]["completion_rates"]),
        }
    return summary


# ── Visualization ─────────────────────────────────────────────────────────────

def _draw_env(ax, env: Environment, position_log=None, step=None):
    rows, cols = env.rows, env.cols

    for r in range(rows):
        for c in range(cols):
            color = "#2c2c2c" if env.grid[r, c] == 1 else "#f0f4f8"
            ax.add_patch(plt.Rectangle((c, rows-1-r), 1, 1,
                         linewidth=0.3, edgecolor="#cfd8dc", facecolor=color))

    for sr, sc in env.charging_stations:
        ax.add_patch(plt.Rectangle((sc+0.1, rows-1-sr+0.1), 0.8, 0.8,
                     facecolor="#FFD600", edgecolor="#F9A825", linewidth=1.2))

    for tr, tc in env.task_locations:
        color = "#A5D6A7" if (tr, tc) in env.completed_tasks else "#EF9A9A"
        ax.add_patch(plt.Rectangle((tc+0.15, rows-1-tr+0.15), 0.7, 0.7,
                     facecolor=color, edgecolor="#888", linewidth=0.8))

    for i in range(env.n_robots):
        if position_log:
            r, c = position_log[i][min(step, len(position_log[i])-1)]
        else:
            r, c = env.robot_positions[i]
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax.plot(c+0.5, rows-1-r+0.5, "o", color=color,
                markersize=11, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.text(c+0.5, rows-1-r+0.5, str(i), ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=6)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_policy_comparison(results: Dict, save_path: str = None):
    """Bar chart comparing policies across metrics."""
    policies = list(results.keys())
    metrics  = ["avg_completion", "avg_reward", "avg_alive"]
    labels   = ["Task Completion Rate", "Avg Total Reward", "Robots Alive (avg)"]
    colors   = ["#1565C0", "#B71C1C", "#2E7D32"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [results[p][metric] for p in policies]
        bars = ax.bar(policies, vals, color=colors, edgecolor="white",
                      linewidth=1.2, width=0.5)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Multi-Robot Policy Comparison under Uncertainty\n"
                 "(averaged over 10 trials with stochastic failures)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_battery_trajectories(result: Dict, env: Environment,
                               save_path: str = None):
    """Plot battery levels over time for each robot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(env.n_robots):
        batt = result["battery_log"][i]
        ax.plot(batt, color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                linewidth=2, label=f"Robot {i}", alpha=0.85)

    ax.axhline(y=6, color="red", linestyle="--", linewidth=1,
               alpha=0.6, label="Recharge threshold")
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Battery level", fontsize=11)
    ax.set_title(f"Battery Trajectories — {result['policy'].capitalize()} Policy\n"
                 f"Tasks completed: {result['tasks_completed']}/{result['tasks_total']}  |  "
                 f"Steps: {result['steps']}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_environment_snapshot(env: Environment, result: Dict,
                               save_path: str = None):
    """Snapshot of environment at start, mid, and end of episode."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    steps = result["steps"]
    snaps = [0, steps//2, steps-1]
    titles = ["Start", "Midpoint", "End"]

    for ax, snap, title in zip(axes, snaps, titles):
        _draw_env(ax, env, result["position_log"], snap)
        ax.set_title(title, fontsize=11, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#FFD600",  label="Charging station"),
        mpatches.Patch(color="#EF9A9A",  label="Pending task"),
        mpatches.Patch(color="#A5D6A7",  label="Completed task"),
        mpatches.Patch(color="#2c2c2c",  label="Obstacle"),
    ] + [mpatches.Patch(color=ROBOT_COLORS[i], label=f"Robot {i}")
         for i in range(min(env.n_robots, 5))]

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Multi-Robot Coordination Snapshots — "
                 f"{result['policy'].capitalize()} Policy",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()