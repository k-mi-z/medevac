from collections import defaultdict
import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t
import sklearn.metrics
import torch
from joblib import Parallel, delayed
from gymnasium.wrappers import FlattenObservation
from algorithms.ddqn import DQN
import gymnasium

class Evaluator():
    def __init__(self, eval_id, env, state_dim, num_actions, n_neurons, device):
        # self.env = gymnasium.make("LunarLander-v3")
        self.env = env.copy()
        self.eval_id = eval_id
        self.device = device
        self.policy_net = DQN(state_dim, num_actions, n_neurons).to(device)

    def evaluate_episode(self, policy_net_state_dict, is_constrained, seed_mult, offset): 
        self.policy_net.load_state_dict(policy_net_state_dict)
        self.policy_net.eval()

        env = FlattenObservation(self.env)

        terminated = False
        truncated = False
        cumulative_reward = 0
        state, info = env.reset(seed=int(seed_mult * 1000 + self.eval_id + offset))

        with torch.no_grad():
            while not(terminated or truncated):
                
                state_action_values = self.policy_net(
                    torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    ).squeeze(0)

                if is_constrained:
                    mask = torch.tensor(info['mask'], dtype=torch.bool, device=self.device)
                    state_action_values[~mask] = float('-inf')
                
                action = state_action_values.argmax().item()

                state, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += reward        

        return cumulative_reward


class EvaluationManager():
    def __init__(
        self, 
        save_path,
        n_neurons,
        device, 
        env,
        is_constrained,
        num_eval_reps,
        n_jobs,
        offset=0,
        num_top_policies=10, 
        ):
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.env = env
        state_dim = FlattenObservation(env).observation_space.shape[0]
        num_actions = env.action_space.n
        self.is_constrained = is_constrained
        self.num_eval_reps = num_eval_reps
        self.n_jobs = n_jobs
        self.offset = offset

        self.evaluators = [
            Evaluator(
                i, 
                env, 
                state_dim, 
                num_actions, 
                n_neurons,
                device
            ) 
            for i in range(2 * num_eval_reps)
        ]

        self.metrics = defaultdict(lambda: {
            'milestones': [], 
            'means': [], 
            'hws': [], 
            'elapsed_times': []
            }
        )

        self.top_policies = [
            {
                'mean': -np.inf, 
                'hw': np.inf, 
                'algo_rep': None, 
                'milestone': None, 
                'policy_net_state_dict': None
            } for _ in range(num_top_policies)
        ]

    @staticmethod
    def evaluate_episode(evaluator, policy_net_state_dict, is_constrained, seed_mult, offset):
        return evaluator.evaluate_episode(policy_net_state_dict, is_constrained, seed_mult, offset)

    def milestone_test(
        self, 
        policy_net_state_dict, 
        algo_rep, 
        milestone, 
        elapsed_time,
        parallelize,
        ):
        seed_mult = 1
        if parallelize:
            with Parallel(n_jobs=self.n_jobs) as parallel:
                cumulative_rewards = parallel(
                    delayed(EvaluationManager.evaluate_episode)(
                        evaluator, policy_net_state_dict, self.is_constrained, seed_mult, self.offset
                    )
                    for evaluator in self.evaluators[:self.num_eval_reps]
                )
        else:
            cumulative_rewards = [
                EvaluationManager.evaluate_episode(
                    evaluator, policy_net_state_dict, self.is_constrained, seed_mult, self.offset
                )
                for evaluator in self.evaluators[:self.num_eval_reps]
            ]
        mean, hw = EvaluationManager.get_mean_hw(cumulative_rewards)
        self.metrics[algo_rep]['milestones'].append(milestone)
        self.metrics[algo_rep]['means'].append(mean)
        self.metrics[algo_rep]['hws'].append(hw)
        self.metrics[algo_rep]['elapsed_times'].append(elapsed_time)
        attained_new_best = False
        for i, policy in enumerate(self.top_policies):
            if mean - hw > policy['mean'] - policy['hw']:
                self.top_policies[i] = {
                    'mean': mean,
                    'hw': hw,
                    'algo_rep': algo_rep,
                    'milestone': milestone,
                    'policy_net_state_dict': policy_net_state_dict,
                }
                attained_new_best = True
                break
        result = {
            'algo_rep': algo_rep,
            'milestone': milestone,
            'mean': mean,
            'hw': hw,
            'elapsed_time': elapsed_time,
        }

        csv_path = self.save_path / 'all_milestone_metrics.csv'
        file_exists = csv_path.exists()
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

    def evaluate_policy_nets(self, path_to_policy_nets, parallelize):
        pattern = re.compile(r'rep(\d+)_episode(\d+)\.pth')

        # Collect files with extracted rep and episode numbers
        files_with_keys = []
        for f in path_to_policy_nets.iterdir():
            if f.is_file() and f.suffix == '.pth':
                match = pattern.search(f.name)
                if match:
                    rep = int(match.group(1))  # Extract rep number
                    episode = int(match.group(2))  # Extract episode number
                    files_with_keys.append((f, rep, episode))

        # Sort files by (rep, episode)
        files_with_keys.sort(key=lambda x: (x[1], x[2]))

        # Process each file in sorted order
        for f, _, _ in files_with_keys:
            data = torch.load(f, map_location="cpu", weights_only=False)
            self.milestone_test(
                data['policy_net_state_dict'],
                data['rep'],
                data['episode'],
                data['elapsed_time'],
                parallelize,
            )
            del data  # Free memory

    def determine_superlative_policy(self, parallelize):
        seed_mult = 2
        means = []
        hws = []
        for policy in self.top_policies:
            if policy['policy_net_state_dict'] is None:
                continue
            if parallelize:
                with Parallel(n_jobs=self.n_jobs) as parallel:
                    cumulative_rewards = parallel(
                        delayed(EvaluationManager.evaluate_episode)(
                            evaluator, policy['policy_net_state_dict'], self.is_constrained, seed_mult, self.offset
                        )
                        for evaluator in self.evaluators
                    )
            else:
                cumulative_rewards = [
                    EvaluationManager.evaluate_episode(
                        evaluator, policy['policy_net_state_dict'], self.is_constrained, seed_mult, self.offset
                    )
                    for evaluator in self.evaluators
                ]
            mean, hw = EvaluationManager.get_mean_hw(cumulative_rewards)
            means.append(mean)
            hws.append(hw)

            result = {
                'algo_rep': policy['algo_rep'],
                'milestone': policy['milestone'],
                'mean': mean,
                'hw': hw,
            }

            csv_path = self.save_path / 'superlative_metrics.csv'
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=result.keys())
                if not csv_path.exists():
                    writer.writeheader()
                writer.writerow(result)

        superlative_index = np.argmax(np.array(means) - np.array(hws)) # superlative 95% confidence interval lower bound
        self.superlative = {
            'policy_net_state_dict': self.top_policies[superlative_index]['policy_net_state_dict'],
            'mean': means[superlative_index],
            'hw': hws[superlative_index],
        }

        torch.save(self.superlative['policy_net_state_dict'], self.save_path / 'superlative.pth')

    @staticmethod
    def get_mean_hw(data, alpha=0.05):
        n = np.size(data) # number of data points
        if n <= 1:
            raise ValueError("At least 2 data points are required to calculate a confidence interval.")
        sample_std = np.std(data, ddof=1) # sample standard deviation (ddof=1)
        se = sample_std / np.sqrt(n) # standard error
        t_score = t.ppf(1 - alpha / 2, n - 1) 
        mean = np.mean(data)
        halfwidth = t_score * se
        return mean, halfwidth

    def determine_summary_statistics(self, parallelize):
        algo_rep_metrics = {
            'max_milestone_means': [],
            'aulcs': [],
            'elapsed_times': [],
        }
        for algo_rep in self.metrics.keys():
            max_milestone_mean = np.max(self.metrics[algo_rep]['means'])
            aulc = sklearn.metrics.auc(self.metrics[algo_rep]['milestones'], self.metrics[algo_rep]['means'])
            elapsed_time = sum(self.metrics[algo_rep]['elapsed_times'])
            algo_rep_metrics['max_milestone_means'].append(max_milestone_mean)
            algo_rep_metrics['aulcs'].append(aulc)
            algo_rep_metrics['elapsed_times'].append(elapsed_time)

        avg_max_milestone_mean, avg_max_milestone_hw = EvaluationManager.get_mean_hw(algo_rep_metrics['max_milestone_means'])
        avg_aulc, avg_aulc_hw = EvaluationManager.get_mean_hw(algo_rep_metrics['aulcs'])
        avg_elapsed_time, avg_elapsed_time_hw = EvaluationManager.get_mean_hw(algo_rep_metrics['elapsed_times'])

        self.determine_superlative_policy(parallelize)

        summary_statistics = pd.DataFrame([
                {
                    'avg_max_milestone_mean': avg_max_milestone_mean,
                    'avg_max_milestone_hw': avg_max_milestone_hw,
                    'avg_aulc': avg_aulc,
                    'avg_aulc_hw': avg_aulc_hw,
                    'avg_elapsed_time': avg_elapsed_time,
                    'avg_elapsed_time_hw': avg_elapsed_time_hw,
                    'superlative_mean': self.superlative['mean'],
                    'superlative_hw': self.superlative['hw'],
                }
            ]
        )
        
        summary_statistics.to_csv(self.save_path / 'summary_statistics.csv', index=False)
        print(summary_statistics)
        

        milestone_means = np.array([self.metrics[algo_rep]['means'] for algo_rep in self.metrics.keys()])
        milestone_hws = np.array([self.metrics[algo_rep]['hws'] for algo_rep in self.metrics.keys()])

        self.avg_milestone_means = np.mean(milestone_means, axis=0)
        self.avg_milestone_hws = np.mean(milestone_hws, axis=0)
        avg_milestone_metrics = pd.DataFrame(
            {
                'milestone': self.metrics[1]['milestones'],
                'avg_mean': self.avg_milestone_means,
                'avg_hw': self.avg_milestone_hws,
            }
        )
        # save avg_milestone_metrics to save_path
        avg_milestone_metrics.to_csv(self.save_path / 'avg_milestone_metrics.csv', index=False)

    def plot_learning_curve(self, x_spacing, yticks, ylim):
        # Plot average milestone means and halfwidths
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(
            self.metrics[1]['milestones'],
            self.avg_milestone_means,
            ms = 3,
            mec = 'k',
            linewidth=1,
        )

        # Plot upper confidence bound
        plt.plot(
            self.metrics[1]['milestones'],
            self.avg_milestone_means + self.avg_milestone_hws,
            color='grey',
            linestyle='--',
            linewidth=0.5,
            label='95% Halfwidth',
        )

        # Plot lower confidence bound
        plt.plot(
            self.metrics[1]['milestones'],
            self.avg_milestone_means - self.avg_milestone_hws,
            color='grey',
            linestyle='--',
            linewidth=0.5,
        )

        plt.xlabel('Episode')
        plt.xticks([milestone for milestone in self.metrics[1]['milestones'] if milestone % x_spacing == 0])
        plt.yticks(yticks)
        plt.ylabel('Mean Cumulative Reward')
        plt.legend(loc="lower right", fontsize=7)
        plt.xlim([-0.05*self.metrics[1]['milestones'][-1], 1.05*self.metrics[1]['milestones'][-1]])
        plt.ylim(ylim)
        ax = plt.gca()  # Get current Axes object
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # save figure to save_path
        plt.savefig(self.save_path / 'learning_curve.pdf', bbox_inches='tight')
        plt.show()