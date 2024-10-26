import torch
import torch.nn as nn
import torch.nn.functional as F


class Prediction(nn.Module):
    def __init__(self, in_feature = 69, hid_units = 256, contract = 1, mid_layers = True, res_con = True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)

        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))

        return out


class ConcurrentQueryLinear(nn.Module):
    def __init__(self, hidden_dim=329, pred_hid=256):
        super(ConcurrentQueryLinear,self).__init__()
        self.linear_time = nn.Linear(1, hidden_dim)
        self.pred = Prediction(hidden_dim, pred_hid)
        
    def forward(self, con_queries, times):      
        time_embedding = self.linear_time(times)
        return self.pred(con_queries + time_embedding).squeeze()
    
class LinearModel(nn.Module):
    def __init__(self, hidden_dim=329, pred_hid=256):
        super(LinearModel,self).__init__()
        self.pred = Prediction(hidden_dim+1, pred_hid)
        
    def forward(self, con_queries, times):
        return self.pred(torch.cat([con_queries, times], dim=2)).squeeze()