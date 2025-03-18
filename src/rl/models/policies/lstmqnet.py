from typing import Any, Dict, List, Optional, Type

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

class LSTMQNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network with LSTM for DQN
    """
    action_space: spaces.Discrete  # Add this line to match QNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,  # Add net_arch parameter
        hidden_size: int = 256,
        lstm_layers: int = 1,
        sequence_length: int = 8,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.features_dim = features_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.sequence_length = sequence_length
        self.activation_fn = activation_fn
        self.net_arch = net_arch if net_arch is not None else [64, 64]
        
        self.obs_buffer = []
        
        self.lstm = nn.LSTM(
            input_size=self.features_dim,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True
        )
        
        self.q_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            self.activation_fn(),
            nn.Linear(self.hidden_size, int(self.action_space.n))
        )
        
        self.hidden = None

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        """
        Forward pass of the network.
        
        :param obs: Current observation
        :return: Q-values for each action
        """
        features = self.extract_features(obs, self.features_extractor)
        sequence = self.get_sequence(features)
        
        # Initialize hidden states if needed
        if self.hidden is None or self.hidden[0].size(1) != sequence.size(0):
            self.reset_hidden_states(sequence.size(0))
        
        # Detach hidden states to prevent backprop through sequence
        hidden_states = (self.hidden[0].detach(), self.hidden[1].detach())
                
        lstm_out, self.hidden = self.lstm(sequence, hidden_states)
        q_values = self.q_net(lstm_out[:, -1, :])
        return q_values

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                hidden_size=self.hidden_size,
                lstm_layers=self.lstm_layers,
                sequence_length=self.sequence_length,
            )
        )
        return data

    def get_sequence(self, features: th.Tensor) -> th.Tensor:
        """
        Creates a sequence of observations for LSTM processing.
        
        :param features: Current observation features tensor of shape (batch_size, features_dim)
        :return: Sequence tensor of shape (batch_size, sequence_length, features_dim)
        """
        batch_size = features.size(0)
        
        # Detach and reshape the features tensor
        features_detached = features.detach()
        
        # Append new observation
        self.obs_buffer.append(features_detached)
        
        # Maintain fixed buffer size
        if len(self.obs_buffer) > self.sequence_length:
            self.obs_buffer = self.obs_buffer[-self.sequence_length:]
        
        # Create padded sequence
        sequence = self.obs_buffer.copy()
        if len(sequence) < self.sequence_length:
            # Pad with zeros at the beginning, matching the batch size
            padding = [th.zeros(batch_size, self.features_dim, device=features.device) 
                    for _ in range(self.sequence_length - len(sequence))]
            sequence = padding + sequence
        
        # Ensure all tensors in sequence have the same batch size
        sequence = [
            s if s.size(0) == batch_size 
            else s.expand(batch_size, -1) if s.size(0) == 1
            else s[:batch_size] if s.size(0) > batch_size
            else th.cat([s, th.zeros(batch_size - s.size(0), s.size(1), device=s.device)], dim=0)
            for s in sequence
        ]
                
        # Stack and create a new computation graph for the sequence
        return th.stack(sequence, dim=1)   

    def reset_hidden_states(self, batch_size: int = 1) -> None:
        device = next(self.parameters()).device
        self.hidden = (
            th.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device),
            th.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        )

    def reset_buffer(self) -> None:
        """Reset the observation buffer."""
        self.obs_buffer = []

    def reset_states(self) -> None:
        """Reset both the hidden states and observation buffer."""
        self.reset_hidden_states()
        self.reset_buffer()
    
class LSTMDQNPolicy(DQNPolicy):
    """
    Policy class for DQN with LSTM
    """
    q_net: LSTMQNetwork
    q_net_target: LSTMQNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        hidden_size: int = 256,
        lstm_layers: int = 1,
        sequence_length: int = 8,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.sequence_length = sequence_length
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def make_q_net(self) -> LSTMQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return LSTMQNetwork(
            **net_args,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            sequence_length=self.sequence_length
        ).to(self.device)
    
    def reset_states(self) -> None:
        """Reset states of both q_net and q_net_target."""
        self.q_net.reset_states()
        self.q_net_target.reset_states()