#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

import sys
import gym
import torch
import importlib

from toolenv import TrainEnv

from agent_config import dqn_config

log_folder = "logs"

def load_agent(name):
    agent_lib = importlib.import_module("agent." + name)
    agent_class = agent_lib.Agent

    return agent_class

def main():
    env = gym.make("LunarLander-v2")
    agent_class = load_agent("dqn_v1")

    agent = agent_class(env.action_space.n, dqn_config)
    agent.load("./dqn_agent_1.pt")
    agent.eval()

    TrainEnv(env, agent, nb_episode=10, render="human").train()

    return 0

if __name__ == '__main__':
    sys.exit(main())
