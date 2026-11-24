"""Experimental GAT-LSTM; requires compatible torch and torch-geometric."""
try:
    import torch
    from torch import nn
    from torch_geometric.nn import GATConv
except ImportError as exc:
    raise ImportError("Install compatible PyTorch and PyTorch Geometric; see requirements-gnn.txt.") from exc

class GATLSTM(nn.Module):
    """Input [batch,time,nodes,features]; output [batch,nodes,output_features]."""
    def __init__(self,input_features,gat_hidden=16,lstm_hidden=32,output_features=2,heads=2,dropout=0.1):
        super().__init__()
        self.gat=GATConv(input_features,gat_hidden,heads=heads,dropout=dropout,concat=True,add_self_loops=True)
        self.dropout=nn.Dropout(dropout)
        self.lstm=nn.LSTM(gat_hidden*heads,lstm_hidden,batch_first=True)
        self.head=nn.Linear(lstm_hidden,output_features)
    def forward(self,x,edge_index):
        batch,time,nodes,features=x.shape; steps=[]
        for t in range(time):
            node_x=x[:,t].reshape(batch*nodes,features)
            edges=torch.cat([edge_index+b*nodes for b in range(batch)],dim=1)
            h=self.dropout(torch.nn.functional.elu(self.gat(node_x,edges)))
            steps.append(h.reshape(batch,nodes,-1))
        seq=torch.stack(steps,dim=1).permute(0,2,1,3).reshape(batch*nodes,time,-1)
        out,_=self.lstm(seq)
        return self.head(out[:,-1]).reshape(batch,nodes,-1)
