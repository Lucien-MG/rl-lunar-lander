#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard

from collections import deque

class Agent:
    """ Initialize agent.
        
    Args:
        nb_actions (int): number of actions available to the agent
        parameters (dict): contain all the parameters needed
    """
    def __init__(self, nb_actions, parameters):
        self._set_parameters(parameters)
        self.nb_actions = nb_actions

        self.primary = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.reset_target()

        self.optimizer = optim.SGD(self.primary.parameters(), lr=self.alpha)

        self.memory = deque(maxlen=self.memory_size)
        self.rewards = deque(maxlen=100)
        self.rewards.append(0)
        self.step_count = 0

        if self.tensorboard_log:
            self.writer = tensorboard.SummaryWriter()

    def _set_parameters(self, configuration):
        self.__dict__ = {k:v for (k,v) in configuration.items()}

    def reset_target(self):
        self.target.load_state_dict(self.primary.state_dict())

    def update_target(self):
        for target_param, primary_param in zip(self.target.parameters(), self.primary.parameters()):
            target_param.data.copy_(self.tau*primary_param.data + (1.0-self.tau)*target_param.data)

    def load(self, path):
        self.primary.load_state_dict(torch.load(path))
        self.reset_target()

    def save(self, path):
        torch.save(self.primary.state_dict(), path)

    def log(self, name, value):
        """ Log the value in function of steps.

        Args:
            name (str): Variable's name.
            value (float): Value to store.
        """
        if self.tensorboard_log:
            self.writer.add_scalar(name, value, self.step_count)

    @torch.no_grad()
    def select_action(self, state) -> int:
        """ Given the state, select an action.

        Args:
            state (obj): the current state of the environment.
        
        Returns:
            action (int): an integer compatible with the task's action space.
        """
        if torch.rand(1).item() > self.epsilon:
            y_pred = self.primary(torch.from_numpy(state).to(self.device))
        else:
            y_pred = torch.rand(self.nb_actions)
        
        action = torch.argmax(y_pred).item()

        return action

    def step(self, state, action, reward, next_state, done) -> None:
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Args:
            state (obj): the previous state of the environment
            action (int): the agent's previous choice of action
            reward (float): last reward received
            next_state (obj): the current state of the environment
            done (bool): whether the episode is complete (True or False)
        """
        self.memory.append((state, action, reward, next_state, done))
        self.rewards[-1] += reward

        if self.step_count % self.update_step == 0:
            self.learn()

        if done:
            if self.step_count > 30000:
                self.epsilon = max(self.epsilon * self.epsilon_decay_factor, self.min_epsilon)
            self.log('Epsilon', self.epsilon)

            if len(self.memory) >= self.batch_size:
                self.log('Reward', torch.mean(torch.tensor(self.rewards)))
                self.rewards.append(0)

        self.step_count += 1

    @torch.no_grad()
    def create_batch(self):
        # Create batch
        random_steps = torch.randint(len(self.memory), (1, self.batch_size))[0]
        states = torch.tensor([self.memory[i][0] for i in random_steps]).to(self.device)
        actions = torch.tensor([self.memory[i][1] for i in random_steps]).to(self.device)
        rewards = torch.tensor([self.memory[i][2] for i in random_steps]).float().to(self.device) / 100
        next_states = torch.tensor([self.memory[i][3] for i in random_steps]).to(self.device)
        dones = torch.tensor([int(self.memory[i][4]) for i in random_steps]).to(self.device)

        return states, actions, rewards, next_states, dones

    def learn(self):
        # Create batch
        states, actions, rewards, next_states, dones = self.create_batch()

        # Calculate target reward and do not forget to detach it from the graph:
        next_state_value = self.target(next_states).max(1)[0].detach() self.pri(next_states).max(1)[0].detach()
        target_reward = (rewards + (self.gamma * next_state_value * (1 - dones)))

        # Actual action values state:
        states_action_values = self.primary(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Error:
        error = torch.mean(torch.pow(states_action_values - target_reward, 2))

        self.log('Loss', error)

        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()

        self.update_target()


class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 48)
        self.fc5 = nn.Linear(48, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc5(x))

        return x
