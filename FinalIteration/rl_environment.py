import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import random
import math
import ast
import os


class DQN(nn.Module):
    # From PyTorch Reinforcement Learning (DQN) Tutorial
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_observations, 2 * num_observations)
        self.layer2 = nn.Linear(2 * num_observations, 2 * num_observations)
        self.layer3 = nn.Linear(2 * num_observations, num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def initialize_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


class DQNEnvironment:
    POLICY_NET_PATH = "C:/Users/Robotics/Desktop/RL/policy_net.pth"
    TARGET_NET_PATH = "C:/Users/Robotics/Desktop/RL/target_net.pth"
    EXPERIENCE_DATA_PATH = "C:/Users/Robotics/Desktop/RL/experience_data.csv"
    METRICS_CSV_PATH = "C:/Users/Robotics/Desktop/RL/training_metrics_rewards.csv"
    S1_METRICS_CSV_PATH = "C:/Users/Robotics/Desktop/RL/training_metrics_state1.csv"
    S2_METRICS_CSV_PATH = "C:/Users/Robotics/Desktop/RL/training_metrics_state2.csv"
    MEMORY_SIZE = 10000
    STEPS_PER_EPOCH = 50

    def __init__(self, training=True):
        self.training = training
        self.memory = {'S1': [0, 0, 0, 0, 0],
                       'action': 0,
                       'reward': 0,
                       'S2': [0, 0, 0, 0, 0]}

        self.policy_net = DQN(num_observations=5, num_actions=5)
        self.target_net = DQN(num_observations=5, num_actions=5)

        # Initialize all necessary files
        if not os.path.exists(self.POLICY_NET_PATH):
            initialize_weights(self.policy_net)
            torch.save(self.policy_net.state_dict(), self.POLICY_NET_PATH)
        else:
            self.policy_net.load_state_dict(torch.load(self.POLICY_NET_PATH))

        if not os.path.exists(self.TARGET_NET_PATH):
            initialize_weights(self.target_net)
            torch.save(self.target_net.state_dict(), self.TARGET_NET_PATH)
        else:
            self.target_net.load_state_dict(torch.load(self.TARGET_NET_PATH))

        if os.path.exists(self.EXPERIENCE_DATA_PATH):
            self.xp = pd.read_csv(self.EXPERIENCE_DATA_PATH)
        else:
            self.xp = pd.DataFrame(columns=['S1', 'action', 'reward', 'S2'])
            self.xp.to_csv(self.EXPERIENCE_DATA_PATH, index=False)

        if os.path.exists(self.METRICS_CSV_PATH):
            metrics = pd.read_csv(self.METRICS_CSV_PATH)
            self.epoch_count = metrics.iloc[-1, 0] + 1
            self.step_count = self.STEPS_PER_EPOCH * self.epoch_count
        else:
            self.step_count = 0
            self.epoch_count = 0
            pd.DataFrame(columns=["epoch", "average_reward", "average_q_value"]).to_csv(self.METRICS_CSV_PATH, index=False)

        if not os.path.exists(self.S1_METRICS_CSV_PATH):
            pd.DataFrame(columns=["epoch", "step", "Tz", "gripper", "tot_angle", "rel_angle", "CT_detected"]).to_csv(self.S1_METRICS_CSV_PATH, index=False)

        if not os.path.exists(self.S2_METRICS_CSV_PATH):
            pd.DataFrame(columns=["epoch", "step", "Tz", "gripper", "tot_angle", "rel_angle", "CT_detected"]).to_csv(self.S2_METRICS_CSV_PATH, index=False)

        self.batch_size = 32
        self.gamma = 0.99
        self.lr = 0.0001
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 5000
        self.epsilon = self.epsilon_start
        self.tau = 0.005

        self.epoch_rewards = []
        self.epoch_q_values = []
        self.epoch_step_count = 0

    def select_action(self):
        """Select action based on current epsilon value if training."""
        network_input = self.get_network_input('S1')
        if self.training:
            if random.random() > self.epsilon:
                # Exploitation
                output = self.policy_net(network_input)
                action = int(torch.argmax(output))
            else:
                # Exploration
                action = random.randint(0, 4)
                print(f'Epsilon: {self.epsilon:.4f} | Random choice due to exploration')
            self.update_epsilon()
        else:
            output = self.policy_net(network_input)
            action = int(torch.argmax(output))
        return action

    def agent(self):
        action = self.select_action()
        self.memory['action'] = action
        return action

    def update_epsilon(self):
        """Decrement epsilon after each training step."""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.step_count / self.epsilon_decay)
                           )
        self.step_count += 1

    def get_network_input(self, state_key):
        """Return tensor of selected state for the network."""
        state = [float(s) for s in self.memory[state_key]]
        return torch.tensor(state, dtype=torch.float32)

    def train_policy_net(self):
        """Train policy net on mini-batch of experience."""
        batch = self.xp.sample(self.batch_size - 1)
        batch.loc[len(batch)] = self.memory
        out_data, target_data = [], []

        for i in range(len(batch)):
            state1_tensor = self.parse_state(batch['S1'].iloc[i])
            q_vals = self.policy_net(state1_tensor)
            out_data.append(q_vals.clone().detach())

            reward, action = int(batch['reward'].iloc[i]), int(batch['action'].iloc[i])
            # Target val is reward + max Q val for next state output by target net
            target_val = reward + self.gamma * float(torch.max(self.target_net(self.parse_state(batch['S2'].iloc[i]))))

            q_target = q_vals.clone()
            q_target[action] = target_val
            target_data.append(q_target)

        avg_q_value = torch.stack(out_data).mean().item()
        self.epoch_q_values.append(avg_q_value)

        self.optimize_model(out_data, target_data)

    def optimize_model(self, out_data, target_data):
        """Compute loss and optimize policy net."""
        out_tensor = torch.stack(out_data)
        target_tensor = torch.stack(target_data)
        loss = self.criterion(out_tensor, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.save(self.policy_net.state_dict(), self.POLICY_NET_PATH)

    def parse_state(self, state_str):
        """Parse and return state tensor from string representation."""
        state = ast.literal_eval(state_str) if isinstance(state_str, str) else state_str
        return torch.tensor(state, dtype=torch.float32)

    def update_tnn(self):
        """Update target network to match policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        torch.save(self.target_net.state_dict(), self.TARGET_NET_PATH)
        self.xp.to_csv(self.EXPERIENCE_DATA_PATH, index=False)

    def update_environment(self):
        reward = 0

        # Check for illegal moves like rotating past boundaries (state won't change in these cases)
        if self.memory['S1'] == self.memory['S2']:
            reward = -3
        
        # Goal-based reward
        elif self.memory['S2'][2] >= 1:
            reward = 100

        # Progress-based rewards
        elif math.isclose((self.memory['S2'][2] - self.memory['S1'][2]), 5/600, rel_tol=1e-7):  # +5 degrees rotated
            reward = 5
        
        elif math.isclose((self.memory['S2'][2] - self.memory['S1'][2]), -5/600, rel_tol=1e-7) and self.memory['S1'][4]:  # -5 degrees rotated, CT detected
            reward = 5
        
        elif (self.memory['S2'][3] - self.memory['S1'][3]) == 120/240 and (self.memory['S2'][2] == self.memory['S1'][2]) and self.memory['S1'][3] <= -1:  # just arm correctly rotated +120
            reward = 5

        elif (self.memory['S2'][3] - self.memory['S1'][3]) == -120/240 and (self.memory['S2'][2] == self.memory['S1'][2]) and self.memory['S1'][3] >= 1:  # just arm correctly rotated -120
            reward = 5
        
        elif self.memory['S2'][4]:  # CT detected
            reward = -3
        
        else:
            reward = 0


        # Save reward
        self.memory['reward'] = reward
        self.epoch_rewards.append(reward)
        print("Reward: ", reward)

        # Log states
        with open(self.S1_METRICS_CSV_PATH, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.epoch_count, self.epoch_step_count, self.memory['S1'][0], self.memory['S1'][1], self.memory['S1'][2], self.memory['S1'][3], self.memory['S1'][4]])
        
        with open(self.S2_METRICS_CSV_PATH, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.epoch_count, self.epoch_step_count, self.memory['S2'][0], self.memory['S2'][1], self.memory['S2'][2], self.memory['S2'][3], self.memory['S2'][4]])

        # End of epoch logging
        if self.epoch_step_count == self.STEPS_PER_EPOCH:
            avg_reward = sum(self.epoch_rewards) / len(self.epoch_rewards)
            avg_q_val = sum(self.epoch_q_values) / len(self.epoch_q_values)
            print('End of epoch')
            print('Average reward: ' + str(avg_reward))
            print('Average Q value: ' + str(avg_q_val))

            with open(self.METRICS_CSV_PATH, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.epoch_count, avg_reward, avg_q_val])

            self.epoch_rewards.clear()
            self.epoch_q_values.clear()

        # Keep experience data to most recent moves
        if len(self.xp) == self.MEMORY_SIZE:
            self.xp.drop(index=0)
            self.xp.reset_index()

        # Update weights of policy QNN
        if len(self.xp) >= self.batch_size and self.training:
            self.train_policy_net()

        # Add current memory to EXP
        self.xp.loc[len(self.xp)] = self.memory
