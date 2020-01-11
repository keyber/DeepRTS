import torch
from torch import nn
import numpy as np

LEFT_CLICK = 1


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, i, min_val, max_val):
        return torch.clamp(i, min_val, max_val)
    
    @staticmethod
    def backward(_ctx, grad_output):
        return grad_output, None, None


clamp = Clamp.apply


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
                w[[i_tile_in  * n + i for i in i_vec_changed], i_tile_out * n + j] =  c
                w[[i_tile_out * n + i for i in i_vec_changed], i_tile_out * n + j] = -c
            
            l[-1].weight.data = w
            l[-1].bias.data = b

        self.layers = nn.Sequential(*l)
    
    def forward(self, x):
        res = self.layers(x)
        
        res = res + x
        
        res[:, :-GameRepresentation.PLAY_DESC] = clamp(res[:, :-GameRepresentation.PLAY_DESC], 0., 1.)
        
        return res


class GameRepresentation:
    COO = np.array([(0, 2),
                            (-1, 1), (0, 1), (1, 1),
                            (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
                            (-1, -1), (0, -1), (1, -1),
                            (0, -2)])
    #COO = np.array([(0, 1), (-1, 0), (0, 0), (1, 0), (0, -1)])
    TILE_DESC = 14
    PLAY_DESC = 4
    
    OUTSIDE = np.array([0, 1] + [0] * (TILE_DESC - 2))  # mur
    
    def __init__(self, map_state, player_state, obtained_from_coo=None, obtained_from_vector=None):
        # if type(player_state) is list:
        #     player_state = torch.tensor(player_state, dtype=torch.int32)
        # assert map_state.dtype == torch.int32, map_state.dtype
        # assert player_state.dtype == torch.int32, player_state.dtype
        self.map_state = map_state
        self.player_state = player_state
        self.obtained_from_coo = obtained_from_coo
        self.obtained_from_vector = obtained_from_vector
    
    @staticmethod
    def create_representation_from_game(game):
        p = game.players[0]
        x, y = game.get_width(), game.get_height()
        s = np.empty((x, y, GameRepresentation.TILE_DESC))
        
        s0 = game.get_state()
        
        # assert np.all((s0[:,:,6]==0) + (s0[:,:,6]==8)), set(list(s0[:, :, 6].reshape(-1)))
        
        # type de case
        s[:, :, 0:5] = s0[:, :, 0].reshape((x, y, 1)) == np.array([2., 3, 4, 5, 6])
        assert np.all(np.sum(s[:, :, 0:5], axis=2) == 1)
        
        # ownership (modified representation)
        s[:, :, 5] = s0[:, :, 1] + s0[:, :, 2] + s0[:, :, 3]
        
        # est un bâtiment
        s[:, :, 6] = s0[:, :, 2]
        
        # est une unité
        s[:, :, 7] = s0[:, :, 3]
        
        # type d'unité/bâtiment
        s[:, :, 8:14] = s0[:, :, 4].reshape((x, y, 1)) == np.array([1., 3, 4, 5, 6, 7])
        
        assert np.all(np.sum(s[:, :, 8:14], axis=2) <= 1)
        
        # vie
        # s[:, :, 14] = s0[:, :, 5]
        
        # état de l'unité
        # s[:, :, 15:] = s0[:, :, 6] == torch.tensor([1, 2, 3, 4, 5, 6, 7, 8],
        #                                            dtype=torch.float32).expand((x, y, 8))
        
        # unité de récolte s0[:, :, 7]
        # s[:, :, [7, 8]] = np.zeros((game.get_width(), game.get_height(), 2))
        
        # attack score s0[:, :, 8]
        # defense score s0[:, :, 9]
        # s[:, :, [9, 10]] = game.get_state()[:, :, [8, 9]]
        
        # for xi in range(s.shape[0]):
        #     for yi in range(s.shape[1]):
        #         if s[xi][yi][1] == 1:
        #             p.do_manual_action(LEFT_CLICK, xi, yi)
        #             u = p.get_targeted_unit()
        #             assert u is not None
        
        # s[yi, xi, [7, 8]] = [u.gold_carry, u.lumber_carry]
        
        p = [p.food, p.food_consumption, p.gold, p.lumber]
        assert len(p) == GameRepresentation.PLAY_DESC
        
        return GameRepresentation(s, p)
    
    @staticmethod
    def get_vector_size():
        # map + player
        return len(GameRepresentation.COO) * GameRepresentation.TILE_DESC + GameRepresentation.PLAY_DESC
    
    def _is_outside(self, x, y):
        return x < 0 or y < 0 or x >= self.map_state.shape[0] or y >= self.map_state.shape[1]
    
    def _get_state_(self, x, y):
        if self._is_outside(x, y):
            return GameRepresentation.OUTSIDE
        return self.map_state[x][y]
    
    def get_vector(self, coo, dtype=torch.float32):
        l = [self._get_state_(x, y) for x, y in GameRepresentation.COO + coo]
        
        res = list(np.stack(l).reshape(-1))
        
        player = self.player_state
        if type(self.player_state) is not list:
            player = list(player)
        res += player
        
        assert len(res) == self.get_vector_size(), (len(res), self.get_vector_size())
        return torch.tensor(res, dtype=dtype)
    
    def create_from_new_vector(self, coo, vector):
        assert vector.dtype == int
        data = (vector.detach() + .5).int()
        
        map_data = data[:-GameRepresentation.PLAY_DESC].reshape((-1, GameRepresentation.TILE_DESC))
        
        play_state = data[-GameRepresentation.PLAY_DESC:]
        
        map_state = self.map_state.copy()
        for x, y in GameRepresentation.COO:
            if not self._is_outside(x + coo[0], y + coo[1]):
                map_state[x + coo[0], y + coo[1]] = map_data[x, y]
        
        return GameRepresentation(map_state, play_state, coo, vector)
    
    def __eq__(self, other):
        return np.all(self.map_state == other.map_state) \
               and all([x == y for x, y in zip(self.player_state, other.player_state)])
    
    def __lt__(self, other):
        return True
    
    def __hash__(self):
        return int(self.map_state.sum() + 97 * sum(self.player_state))


class WorldModel(nn.Module):
    def __init__(self, state_space, action_space, layers):
        super().__init__()
        
        one_action_net = {a: OneActionNet(state_space, layers) for a in action_space}
        
        self.nets = nn.ModuleDict(one_action_net)
    
    def forward_full(self, game_repres, pos, action):
        v_out = self.forward_vec(action, game_repres, pos)
        
        return game_repres.create_from_new_vector(pos, v_out)
    
    def forward_vec(self, action, game_repres=None, pos=None, v_in=None):
        if v_in is None:
            v_in = game_repres.get_vector(pos)
        
        # noinspection PyUnresolvedReferences
        return self.nets[action](v_in)


def pprint(game_repr, p):
    for i in range(GameRepresentation.TILE_DESC):
        print(set([s[i] for s in game_repr.map_state.reshape((-1, GameRepresentation.TILE_DESC))]))
    print("p:", p.food, p.food_consumption, p.gold, p.lumber)
    
    for xi in range(game_repr.map_state.shape[0]):
        for yi in range(game_repr.map_state.shape[1]):
            if game_repr.map_state[xi][yi][1] == 1:
                print((xi, yi))
    u = p.get_targeted_unit()
    if u is None:
        print("u: No Selection")
    else:
        print("u", u.health, u.gold_carry, u.lumber_carry)
