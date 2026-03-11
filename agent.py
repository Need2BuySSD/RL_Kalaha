import torch
import numpy as np


class AgentNetwork(torch.nn.Module):
	def __init__(self, input_size=14, N_actions=6, hidden_size=128):
		super().__init__()
		self.backbone = torch.nn.Sequential(
			torch.nn.Linear(input_size, 2*hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(2*hidden_size, hidden_size),
			torch.nn.ReLU(),
		)
		self.actor = torch.nn.Sequential(
			torch.nn.Linear(hidden_size, hidden_size // 2),
			torch.nn.Tanh(),
			torch.nn.Linear(hidden_size // 2, N_actions)
		)
        
		self.critic = torch.nn.Sequential(
			torch.nn.Linear(hidden_size, hidden_size // 2),
			torch.nn.Tanh(),
			torch.nn.Linear(hidden_size // 2, 1)
		)
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, torch.nn.Linear):
			torch.nn.init.orthogonal_(module.weight, gain=1.0)
			torch.nn.init.constant_(module.bias, 0.0)
		elif isinstance(module, torch.nn.BatchNorm1d):
			torch.nn.init.constant_(module.weight, 1)
			torch.nn.init.constant_(module.bias, 0)
			

	def forward(self, state):
		state_features = self.backbone(state)
		return self.actor(state_features), self.critic(state_features).squeeze(-1)
	


class A2CAgent:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def act(self, inputs, mask=None, greedy=False, epsilon=.15, eps_greedy=True):
        inputs = torch.tensor(inputs, dtype=torch.float32 ).to(self.device)
        action_logits, state_values = self.model(inputs)

        if mask is None:
            mask = torch.ones_like(action_logits).to(self.device)

        masked_action_probs = torch.softmax(action_logits, dim=-1) * mask
        masked_action_probs /= (1e-10 + masked_action_probs.sum(dim=-1))
        
        action_dist = torch.distributions.Categorical(probs=masked_action_probs)
        entropy = action_dist.entropy()

        if greedy:
            actions = torch.argmax(masked_action_probs, dim=-1).item()
            log_probs = torch.log(masked_action_probs[actions])
        elif eps_greedy:
            actions = action_dist.sample().item()
            log_probs = torch.log(masked_action_probs[actions]) 
        else:
            if np.random.sample() > epsilon:
                actions = torch.argmax(masked_action_probs, dim=-1).item()
            else:
                W = mask.cpu().numpy()
                actions = np.random.choice(np.arange(N_ACTIONS), p=W/W.sum())
            log_probs = torch.log(masked_action_probs[actions]) 

        assert not torch.isnan(log_probs).any()
        return actions, log_probs, state_values, entropy