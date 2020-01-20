import torch
from torch import nn
from worldModel import GameRepresentation
import sklearn.tree
import pickle


class ValueModel(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        
        self.nets = nn.ModuleDict({a: nn.Linear(state_space, 1) for a in action_space})
    
    def forward_vec(self, action, game_repres=None, v_in=None, check=False):
        if v_in is None:
            v_in = game_repres.get_vector_full()
        
        ndim = v_in.ndim
        if ndim == 1:
            v_in = v_in.unsqueeze(0)
        
        if check:
            GameRepresentation.is_correct_vector(v_in, raise_=True)
        
        # noinspection PyUnresolvedReferences
        res = self.nets[action](v_in)
        
        GameRepresentation.is_correct_vector(res, raise_=check)
        
        if ndim == 1:
            res = res.squeeze(0)
        
        return res


class ValueModel2(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.layer = nn.Linear(state_space, 1)
    
    def forward(self, x):
        return self.layer(x).squeeze()


class RewardModel2:
    def __init__(self):
        self.tree = sklearn.tree.DecisionTreeRegressor()  # type: sklearn.tree.DecisionTreeRegressor 
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x1, x2):
        ndim = x1.ndim
        if ndim == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
        
        x = torch.cat((x1, x2), dim=1)
        
        res = self.tree.predict(x)
        
        # print(x.shape, res.shape)
        # res = res.squeeze(1)
        
        if ndim == 1:
            res = res.squeeze(0)
        
        return res
    
    def fit(self, x, y):
        x = torch.cat(x, dim=1)
        
        self.tree.fit(x.numpy(), y.numpy())
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.tree, f)
    
    def load(self, path):
        with open(path, "rb") as f:
            self.tree = pickle.load(f)


class RewardModel(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(2 * state_space),
            nn.Linear(2 * state_space, 100),
            nn.Sigmoid(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x1, x2):
        ndim = x1.ndim
        if ndim == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
        
        x = torch.cat((x1, x2), dim=1)
        res = self.layers(x)
        
        res = res.squeeze(1)
        
        if ndim == 1:
            res = res.squeeze(0)
        
        return res
