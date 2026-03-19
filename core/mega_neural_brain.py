import torch
import torch.nn as nn

class MegaStrategyNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=4):
        super(MegaStrategyNet, self).__init__()
        
        # INCREASED DROPOUT TO 0.5: Forces the network to generalize, not memorize
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.5 
        )
        
        self.regime_gate = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4), # Added Dropout
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
        
        # Added Dropout to the Experts
        self.expert_theta = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
        self.expert_vega  = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
        self.expert_bull  = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
        self.expert_bear  = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        
        regime_probs = self.regime_gate(out) 
        
        out_0 = self.expert_theta(out)
        out_1 = self.expert_vega(out)
        out_2 = self.expert_bull(out)
        out_3 = self.expert_bear(out)
        
        experts_stacked = torch.stack([out_0, out_1, out_2, out_3], dim=1)
        regime_probs_exp = regime_probs.unsqueeze(2) 
        
        final_output = torch.sum(experts_stacked * regime_probs_exp, dim=1)
        return final_output