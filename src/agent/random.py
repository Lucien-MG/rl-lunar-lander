#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import numpy as np

class Agent:
    """ Initialize agent.
        
    Args:
        nb_actions (int): number of actions available to the agent
    """
    def __init__(self, nb_actions):
        self.nb_actions = nb_actions

    def select_action(self, state) -> int:
        """ Given the state, select an action.

        Args:
            state (obj): the current state of the environment.
        
        Returns:
            action (int): an integer compatible with the task's action space.
        """
        return np.random.choice(self.nb_actions)

    def step(self, state, action, reward, next_state, done) -> None:
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Args:
            state (obj): the previous state of the environment
            action (int): the agent's previous choice of action
            reward (float): last reward received
            next_state (obj): the current state of the environment
            done (bool): whether the episode is complete (True or False)
        """
        return
