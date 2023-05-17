# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch
from ..net import GATConvNet, GCNConvNet, ResNetBlock, MLPNet


class ActorCritic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim)
        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim)
        self._last_hidden_state = None

    def act(self, obs):
        logits = self.actor(obs)
        return logits

    def evaluate(self, obs):
        value = self.critic(obs)
        return value

    def get_last_rnn_state(self):
        return self._last_hidden_state

    def set_last_rnn_hidden(self, hidden_state):
        self._last_hidden_state = hidden_state


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64):
        super(Actor, self).__init__()
        self.net = GNN(p_net_num_nodes, p_net_feature_dim, embedding_dim=embedding_dim)

    def forward(self, obs):
        """Return logits of actions"""
        logits = self.net(obs)
        return logits


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=64):
        super(Critic, self).__init__()
        self.net = GNN(p_net_num_nodes, p_net_feature_dim, embedding_dim=embedding_dim)

    def forward(self, obs):
        """Return logits of actions"""
        logits = self.net(obs)
        value = torch.mean(logits, dim=-1, keepdim=True)
        return value


class GNN(nn.Module):

    def __init__(self, p_net_num_nodes, feature_dim, embedding_dim=64):
        super(GNN, self).__init__()
        self.gcn = GCNConvNet(feature_dim, embedding_dim, embedding_dim=embedding_dim, dropout_prob=0.,
                              return_batch=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten()
        )

    def forward(self, obs):
        batch_p_net = obs['p_net']  # 100,5
        p_node_embeddings = self.gcn(batch_p_net)  # 1,100,128
        p_node_embeddings = p_node_embeddings.reshape(batch_p_net.num_graphs, -1, p_node_embeddings.shape[-1])
        logits = self.mlp(p_node_embeddings)  # 1,100
        return logits
