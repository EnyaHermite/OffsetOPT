import torch
from S1.BaseNet import S1BaseNet


class S2ReconNet(S1BaseNet):
    def __init__(self, Cin, knn=50, Lembed=8):
        super().__init__(Cin, knn)        
        self.Lembed = Lembed
        self.resolution = 0.01

    def _posEnc(self, x):
        # Create frequency scales in a single operation
        levels = 2. ** torch.arange(self.Lembed, dtype=x.dtype, device=x.device)
        levels = levels.view(1, 1, -1, 1)
        level_x = x[...,None,:]*levels

        # Concatenate sine and cosine functions in a vectorized way
        pe_feats = torch.stack([torch.sin(level_x), torch.cos(level_x)], dim=3)
        pe_feats = pe_feats.view(x.shape[0], x.shape[1], -1)
        pe_feats = torch.cat([x, pe_feats], dim=-1)
        return pe_feats

    def _updateFeatures(self, features):
        knn_dists = torch.linalg.norm(features, axis=-1)
        knn_scale = knn_dists[:,1:2]
        features = self.resolution*features/knn_scale[...,None]   
        features = self._posEnc(features)        # high-resolution signals
        return features

    def forward(self, features, offsets_features):
        in_feats = self._updateFeatures(features+offsets_features)

        # offset predictions
        x = self.layer0(in_feats)
        x = x + self.transformer_posenc  
        x = self.transformer_encoder(x)
        x = x.flatten(start_dim=1)
        logits = self.predict(x)
        
        # enforce the logits prediction to be symmetric
        logits = logits.reshape([-1,(self.K-1),(self.K-1)])
        logits = (logits + logits.permute(0,2,1))
        logits = logits.reshape([-1,(self.K-1)*(self.K-1)])       
        return logits



if __name__=='__main__':
    N, K, C = 400, 50, 99  # K is the sequence length, C is the embedding dimension
    model = S2ReconNet(Cin=C, knn=K)
    
    # Example input of shape (1, K, C)
    input_tensor = torch.randn(N, K, C)
    
    # Forward pass
    offsets, output = model(input_tensor)
    print(offsets.shape, output.shape)  # Should be (1, K, C)
    