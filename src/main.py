"""
Main experiment runner — Multi-Robot Planning under Uncertainty.
Runs all experiments and generates result plots.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from environment import Environment, UncertaintyModel
from simulation import (run_episode, compare_policies,
                         plot_policy_comparison,
                         plot_battery_trajectories,
                         plot_environment_snapshot)

os.makedirs("../results", exist_ok=True)

if __name__ == "__main__":
    print("="*60)
    print("  MULTI-ROBOT PLANNING UNDER UNCERTAINTY")
    print("  Mission Sudarshan Chakra — UAV Coordination")
    print("="*60)

    uncertainty = UncertaintyModel(
        action_failure_prob=0.10,
        battery_spike_prob=0.05,
        comm_loss_prob=0.08
    )

    env = Environment(
        rows=10, cols=10,
        n_robots=3,
        battery_capacity=20,
        obstacle_ratio=0.15,
        n_tasks=6,
        uncertainty=uncertainty,
        seed=42
    )
    print(f"\n{env}")
    print(f"Charging stations: {env.charging_stations}")
    print(f"Tasks: {env.task_locations}")

    # ── Experiment 1: Single episode comparison ──
    print("\n── Experiment 1: Single Episode per Policy ──")
    for policy in ["random", "greedy", "coordinated"]:
        r = run_episode(env, policy=policy, max_steps=200, verbose=True)
        print(f"\n[{policy.upper()}]")
        print(f"  Tasks completed : {r['tasks_completed']}/{r['tasks_total']}")
        print(f"  Steps taken     : {r['steps']}")
        print(f"  Total reward    : {r['total_reward']:.1f}")
        print(f"  Robots alive    : {r['robots_alive']}/{env.n_robots}")
        print(f"  Final batteries : {r['final_batteries']}")

    # ── Experiment 2: Policy comparison over 10 trials ──
    print("\n── Experiment 2: Policy Comparison (10 trials) ──")
    summary = compare_policies(env, n_trials=10)
    for policy, stats in summary.items():
        print(f"\n[{policy.upper()}]")
        for k, v in stats.items():
            print(f"  {k:20s}: {v:.3f}")

    plot_policy_comparison(summary, save_path="../results/policy_comparison.png")

    # ── Experiment 3: Battery trajectories ──
    print("\n── Experiment 3: Battery Trajectories ──")
    env.uncertainty = UncertaintyModel(seed=42)
    result_coord = run_episode(env, policy="coordinated", max_steps=200)
    plot_battery_trajectories(result_coord, env,
                               save_path="../results/battery_trajectories.png")

    # ── Experiment 4: Environment snapshots ──
    print("\n── Experiment 4: Environment Snapshots ──")
    env.uncertainty = UncertaintyModel(seed=42)
    result_coord2 = run_episode(env, policy="coordinated", max_steps=200)
    plot_environment_snapshot(env, result_coord2,
                               save_path="../results/env_snapshots.png")

    print("\n✓ All experiments done. Results saved to ../results/")