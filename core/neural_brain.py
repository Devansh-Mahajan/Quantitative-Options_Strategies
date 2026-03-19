import torch
import torch.nn as nn
import logging

logger = logging.getLogger(f"strategy.{__name__}")

class StrategySelectorNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(StrategySelectorNet, self).__init__()
        
        # 1. The Deep Sequence Encoder (Reads 60 days of macro data)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3
        )
        
        # 2. The "Deep HMM" Gating Network (Predicts the Market Cycle)
        self.regime_gate = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1) # Outputs % probability of Calm, Choppy, Panic
        )
        
        # 3. The "Experts" (Finding the Options Edge)
        # Each expert gets its own dedicated dense layers to find specific advantages
        self.expert_theta = nn.Sequential(nn.Linear(hidden_size, 32), nn.GELU(), nn.Linear(32, num_classes))
        self.expert_vega  = nn.Sequential(nn.Linear(hidden_size, 32), nn.GELU(), nn.Linear(32, num_classes))
        self.expert_hedge = nn.Sequential(nn.Linear(hidden_size, 32), nn.GELU(), nn.Linear(32, num_classes))

    def forward(self, x):
        # Pass the 60-day sequence through the LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Grab the final mathematical "thought" at the end of the 60 days
        
        # 1. Calculate Market Cycle Probabilities (The HMM part)
        # Example output: [0.80, 0.15, 0.05] (80% Calm, 15% Choppy, 5% Panic)
        regime_probs = self.regime_gate(out) 
        
        # 2. Each expert calculates the expected edge for its specific strategy
        out_theta = self.expert_theta(out)
        out_vega = self.expert_vega(out)
        out_hedge = self.expert_hedge(out)
        
        # Stack the experts together: Shape [Batch, 3, Num_Classes]
        experts_stacked = torch.stack([out_theta, out_vega, out_hedge], dim=1)
        
        # 3. Institutional Blending
        # Multiply each expert's prediction by the Regime probability.
        # If the network is 99% sure it's a Panic regime, the Hedge Expert's prediction 
        # is multiplied by 0.99, completely overriding the Theta Expert.
        regime_probs = regime_probs.unsqueeze(2) 
        final_output = torch.sum(experts_stacked * regime_probs, dim=1)
        
        return final_output