import torch
import torch.nn as nn


class S1BaseNet(nn.Module):
    def __init__(self, Cin, knn=50):
        super().__init__()
        
        self.K = knn
        self.BC = 64
        self.layer0 = nn.Linear(Cin, self.BC)  
              
        self.transformer_posenc = nn.Parameter(torch.zeros(1, knn, self.BC))
        EncLayer = nn.TransformerEncoderLayer(d_model=self.BC, 
                            nhead=4, dim_feedforward=256,                                     
                            dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(EncLayer, num_layers=5)                  
        self.predict = nn.Linear(knn*self.BC, (knn-1)*(knn-1))

    def forward(self, in_feats):
        # offset predictions
        x = self.layer0(in_feats)
        x = x + self.transformer_posenc # self.nn_posenc 
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
    model = S1BaseNet(Cin=C, knn=K, num_layers=5)
    
    # Example input of shape (1, K, C)
    input_tensor = torch.randn(N, K, C)
    
    # Forward pass
    logits = model(input_tensor)
    print(logits.shape, (K-1)**2)  # Should be (1, K, C)
    