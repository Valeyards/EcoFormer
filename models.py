# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EcoFormer(nn.Module):
    def __init__(self, input_dims, hidden_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_projections = nn.ModuleDict({
            'mrna': nn.Linear(input_dims['mrna'], hidden_dim),
            'lncrna': nn.Linear(input_dims['lncrna'], hidden_dim),
            'circrna': nn.Linear(input_dims['circrna'], hidden_dim)
        })
        
        self.shared_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.shared_norm1 = nn.LayerNorm(hidden_dim)
        self.shared_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.shared_norm2 = nn.LayerNorm(hidden_dim)
        self.silu = nn.SiLU()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.expert_nets = nn.ModuleDict({
            'mrna': self._make_expert_net(hidden_dim),
            'lncrna': self._make_expert_net(hidden_dim),
            'circrna': self._make_expert_net(hidden_dim)
        })
        
        self.proj = nn.Linear(hidden_dim * 3, 2)
    
    def _make_expert_net(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, data):
        mrna = data['mrna']
        lncrna = data['lncrna']
        circrna = data['circrna']
        rna_features = {}
        inputs = {
            'mrna': mrna,
            'lncrna': lncrna,
            'circrna': circrna
        }
        
        for rna_type, x in inputs.items():
            x = self.input_projections[rna_type](x)
            identity = x
            x = self.shared_linear1(x)
            x = self.silu(x)
            x = self.shared_norm1(x)
            x = self.shared_linear2(x)
            x = self.silu(x)
            x = self.shared_norm2(x)
            x = x + identity
            
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x_with_cls = torch.cat([cls_tokens, x.unsqueeze(1)], dim=1)
            
            shared_identity = x_with_cls
            shared_out = self.shared_transformer(x_with_cls)
            shared_out = shared_out + shared_identity
            
            cls_feature = shared_out[:, 0]
            mean_feature = shared_out[:, 1:].mean(dim=1)
            shared_feature = cls_feature + mean_feature
            
            expert_out = self.expert_nets[rna_type](shared_feature)
            expert_out = expert_out + shared_feature
            
            rna_features[rna_type] = expert_out
        
        combined_features = torch.cat([
            rna_features['mrna'],
            rna_features['lncrna'],
            rna_features['circrna']
        ], dim=1)
        
        return self.proj(combined_features)

def get_model(model_name, input_dims, **kwargs):
    if model_name == 'ecoformer':
        transformer_params = {
            'input_dims': input_dims,
            'hidden_dim': kwargs.get('hidden_dim', 512),
            'num_heads': kwargs.get('num_heads', 8),
            'num_layers': kwargs.get('num_layers', 2),
            'dropout': kwargs.get('dropout', 0.1)
        }
        return EcoFormer(**transformer_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
