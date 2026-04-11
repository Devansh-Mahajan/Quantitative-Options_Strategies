import torch
import torch.nn as nn


class MegaStrategyNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.35,
        )

        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.regime_gate = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

        self.expert_theta = nn.Sequential(
            nn.Linear(hidden_size, 96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )
        self.expert_vega = nn.Sequential(
            nn.Linear(hidden_size, 96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )
        self.expert_bull = nn.Sequential(
            nn.Linear(hidden_size, 96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )
        self.expert_bear = nn.Sequential(
            nn.Linear(hidden_size, 96),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        seq_out, _ = self.lstm(x)

        attn_logits = self.temporal_attention(seq_out).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        context = torch.sum(seq_out * attn_weights, dim=1)

        regime_probs = self.regime_gate(context)

        out_0 = self.expert_theta(context)
        out_1 = self.expert_vega(context)
        out_2 = self.expert_bull(context)
        out_3 = self.expert_bear(context)

        experts_stacked = torch.stack([out_0, out_1, out_2, out_3], dim=1)
        regime_probs_exp = regime_probs.unsqueeze(2)

        final_output = torch.sum(experts_stacked * regime_probs_exp, dim=1)
        return final_output
