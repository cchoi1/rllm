import hashlib
import json
import os
from collections import defaultdict

import torch


def compute_pass_at_k(results):
    # Create a map to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        task = trajectory.task

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        is_correct = 1 if trajectory.reward > 0 else 0

        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)


def compute_pass_at_k_per_turn(results):
    """
    Compute cumulative pass@1 for each turn across all trajectories.
    Cumulative means: if a problem is solved at turn N, it counts as passed
    for all turns >= N.
    
    Args:
        results: List of Trajectory objects
        
    Returns:
        Dictionary mapping turn number to cumulative pass@1 value
    """
    # Track when each trajectory/problem was first solved
    trajectory_solved_at_turn = []
    
    for trajectory in results:
        # Find the first turn where this trajectory was solved
        solved_at_turn = None
        for step_idx, step in enumerate(trajectory.steps):
            turn_num = step_idx + 1
            
            # Check if solved at this turn
            raw_obs = step.info.get('raw_observation') if step.info else None
            if raw_obs and isinstance(raw_obs, dict):
                is_solved = bool(raw_obs.get("solved", False))
            elif step.observation and isinstance(step.observation, dict):
                is_solved = bool(step.observation.get("solved", False))
            else:
                is_solved = False
            
            if is_solved and solved_at_turn is None:
                solved_at_turn = turn_num
                break
        
        trajectory_solved_at_turn.append(solved_at_turn)
    
    # Find the maximum turn number across all trajectories
    max_turn = 0
    for trajectory in results:
        max_turn = max(max_turn, len(trajectory.steps))
    
    # Calculate cumulative pass@1 for each turn
    total_problems = len(results)
    pass_at_k_per_turn = {}
    
    for turn_num in range(1, max_turn + 1):
        # Count how many problems were solved by this turn (cumulative)
        problems_solved_by_turn = sum(
            1 for solved_turn in trajectory_solved_at_turn
            if solved_turn is not None and solved_turn <= turn_num
        )
        
        if total_problems > 0:
            pass_at_k_per_turn[turn_num] = problems_solved_by_turn / total_problems
        else:
            pass_at_k_per_turn[turn_num] = 0.0
    
    return pass_at_k_per_turn


def print_pass_at_k_per_turn(results):
    """
    Print pass@1 for each turn.
    
    Args:
        results: List of Trajectory objects
    """
    pass_at_k_per_turn = compute_pass_at_k_per_turn(results)
    
    print("\n" + "="*50)
    print("Pass@1 per Turn:")
    print("="*50)
    for turn_num in sorted(pass_at_k_per_turn.keys()):
        pass_at_1 = pass_at_k_per_turn[turn_num]
        print(f"Turn {turn_num}: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")
    print("="*50 + "\n")


def save_trajectories(results, save_dir="./trajectories", filename="trajectories.pt"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(results, save_path)
    print(f"Trajectories saved to {save_path}")
    return save_path
