# ==============================================================================
# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

from base import Solution, SolutionStepEnvironment
from solver import registry
from .sub_env import SubEnv
from .net import ActorCritic, Actor, Critic
from ..buffer import RolloutBuffer
from ..rl_solver import RLSolver, PPOSolver, InstanceAgent, A2CSolver
from ..utils import get_pyg_data


@registry.register(
    solver_name='pg_gnn',
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning')
class PgGnnSolver(InstanceAgent, A2CSolver):
    """
    A Reinforcement Learning-based solver that uses
    Advantage Actor-Critic (A3C) as the training algorithm,
    and Graph Convolutional Network (GCN) as the neural network model.
    """

    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self)
        A2CSolver.__init__(self, controller, recorder, counter, **kwargs)
        num_p_net_nodes = kwargs['p_net_setting']['num_nodes']
        self.policy = ActorCritic(p_net_num_nodes=num_p_net_nodes, p_net_feature_dim=5, v_net_feature_dim=2,
                                  embedding_dim=self.embedding_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic},
        ],
        )
        self.SubEnv = SubEnv
        self.preprocess_obs = obs_as_tensor
        self.compute_advantage_method = 'mc'

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.SubEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        sub_obs = sub_env.get_observation()
        sub_done = False
        while not sub_done:
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            tensor_sub_obs = self.preprocess_obs(sub_obs, device=self.device)
            action, action_logprob = self.select_action(tensor_sub_obs, mask=mask, sample=True)
            next_sub_obs, sub_reward, sub_done, sub_info = sub_env.step(action[0])

            if sub_done:
                break

            sub_obs = next_sub_obs
        return sub_env.solution

    def learn_with_instance(self, instance):
        # sub env for sub agent
        sub_buffer = RolloutBuffer()
        v_net, p_net = instance['v_net'], instance['p_net']
        sub_env = self.SubEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
        sub_obs = sub_env.get_observation()
        sub_done = False
        while not sub_done:
            mask = np.expand_dims(sub_env.generate_action_mask(), axis=0)
            tensor_sub_obs = self.preprocess_obs(sub_obs, device=self.device)
            action, action_logprob = self.select_action(tensor_sub_obs, mask=mask, sample=True)
            value = self.estimate_obs(tensor_sub_obs) if hasattr(self.policy, 'evaluate') else None
            next_sub_obs, sub_reward, sub_done, sub_info = sub_env.step(action[0])

            sub_buffer.add(sub_obs, action, sub_reward, sub_done, action_logprob, value=value)
            sub_buffer.action_masks.append(mask)

            if sub_done:
                break

            sub_obs = next_sub_obs

        last_value = self.estimate_obs(self.preprocess_obs(next_sub_obs, self.device)) if hasattr(self.policy,                                                                                  'evaluate') else None
        solution = sub_env.solution
        return solution, sub_buffer, last_value


def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'])
        obs_p_net = Batch.from_data_list([data]).to(device)
        # obs_p_node_id = torch.LongTensor([obs['p_node_id']]).to(device)
        # obs_hidden_state = torch.FloatTensor(obs['hidden_state']).unsqueeze(dim=0).to(device)
        # obs_encoder_outputs = torch.FloatTensor(obs['encoder_outputs']).unsqueeze(dim=0).to(device)
        return {'p_net': obs_p_net, 'mask': None}
    # batch
    elif isinstance(obs, list):
        obs_batch = obs
        p_net_data_list= []
        for observation in obs_batch:
            p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'])
            p_net_data_list.append(p_net_data)
        obs_p_net = Batch.from_data_list(p_net_data_list).to(device)
        # Pad sequences with zeros and get the mask of padded elements
        return {'p_net': obs_p_net, 'mask': None}
    else:
        raise ValueError('obs type error')