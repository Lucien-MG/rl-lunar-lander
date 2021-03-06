#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

dqn_config = {
    "alpha": 1e-2,
    "gamma": 0.99,
    "epsilon": 1.0,
    "min_epsilon": 1e-2,
    "epsilon_decay_factor": 0.999,
    "memory_size": 50000,
    "batch_size": 128,
    "update_step": 8,
    "tau": 1e-3, 
    "device": "cpu",
    "tensorboard_log": True
}
