import torch
from torch import nn
from GameRepresentation import GameRepresentation, tens

LEFT_CLICK = 1


# noinspection PyMethodOverriding
class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, i, min_val, max_val):
        return torch.clamp(i, min_val, max_val)
    
    @staticmethod
    def backward(_ctx, grad_output):
        return grad_output, None, None


class OneActionNet(nn.Module):
    def __init__(self, state_space, layers, c=None):
        super().__init__()
        
        l = []
        size = state_space
        l.append(nn.BatchNorm1d(size))
        for x in layers:
            l.append(nn.Linear(size, x))
            l.append(nn.ReLU())
            size = x
        l.append(nn.Linear(size, state_space))
        
        # sigmoid done in loss
        # no activation for regression
        
        if c is not None:
            w = torch.zeros_like(l[-1].weight)
            b = torch.zeros_like(l[-1].bias)
            
            i_tile_in = 2
            i_tile_out = 4
            
            n = GameRepresentation.TILE_DESC
            
            i_vec_check = [1, 2, 3, 4, 6, 7]
            i_vec_changed = [5, 7, 8]
            
            b[[i_tile_in * n + i for i in i_vec_changed]] = -c
            b[[i_tile_out * n + i for i in i_vec_changed]] = c
            
            for j in i_vec_check:
                w[[i_tile_in * n + i for i in i_vec_changed], i_tile_out * n + j] = c
                w[[i_tile_out * n + i for i in i_vec_changed], i_tile_out * n + j] = -c
            
            l[-1].weight.data = w
            l[-1].bias.data = b
        
        self.layers = nn.Sequential(*l)
    
    def forward(self, x):
        res = self.layers(x)
        
        res = res + x
        
        # noinspection PyUnresolvedReferences
        # res[:, :-GameRepresentation.PLAY_DESC] = Clamp.apply(res[:, :-GameRepresentation.PLAY_DESC], 0., 1.)
        
        return res


class WorldModel(nn.Module):
    def __init__(self, state_space, action_space, layers):
        super().__init__()
        
        one_action_net = {a: OneActionNet(state_space, layers) for a in action_space}
        
        self.nets = nn.ModuleDict(one_action_net)
    
    def forward_full(self, game_repres, pos, action):
        assert not self.training
        
        v_out = self.forward_vec(action, game_repres, pos, check=GameRepresentation.DEBUG)
        return game_repres.create_from_new_vector(pos, v_out)
    
    def forward_vec(self, action, game_repres=None, pos=None, v_in=None, check=False):
        if v_in is None:
            v_in = game_repres.get_vector(pos)
        
        ndim = v_in.ndim
        if ndim == 1:
            v_in = v_in.unsqueeze(0)
        
        if check:
            GameRepresentation.is_correct_vector(v_in, raise_=True)
        
        # noinspection PyUnresolvedReferences
        res = self.nets[action](v_in)
        
        if check:
            check = GameRepresentation.is_correct_vector(res)
            if not check[0]:
                print("forward vec error")
                print(check[1].args)
                
                print(tens(v_in[0, :-GameRepresentation.PLAY_DESC].reshape((-1, GameRepresentation.TILE_DESC))))
                print(v_in[0, -GameRepresentation.PLAY_DESC:])
                
                print(tens(res[0, :-GameRepresentation.PLAY_DESC].reshape((-1, GameRepresentation.TILE_DESC))))
                print(res[0, -GameRepresentation.PLAY_DESC:])
                GameRepresentation.is_correct_vector(res, raise_=True)
        
        if ndim == 1:
            res = res.squeeze(0)
        
        return res
