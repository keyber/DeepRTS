import torch
from torch import nn
import numpy as np

LEFT_CLICK = 1


class OneActionNet(nn.Module):
    def __init__(self, map_space, player_space, layers):
        super().__init__()
        self.n_tiles, self.tile_size = map_space
        
        l = []
        size = self.n_tiles*self.tile_size + player_space
        for x in layers:
            # l.append(nn.BatchNorm1d(size)) #todo 
            l.append(nn.Linear(size, x))
            l.append(nn.ReLU())
            size = x
        
        self.linear_map = nn.Linear(size, map_space)
        # self.map_act = nn.Sigmoid() # fait dans la loss
        
        self.linear_player = nn.Linear(size, player_space)  # no activation for regression
        
        self.layers = nn.Sequential(*l)
    
    def forward(self, x):
        b = x.shape[0]
        hid = self.layers(x)
        map_ = self.map_act(self.linear_map(hid))
        play = self.linear_player(hid)
        return map_.reshape((b, self.n_tiles, self.tile_size)), play


class GameRepresentation:
    COO = np.array([(0, 2),
                    (-1, 1), (0, 1), (1, 1),
                    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
                    (-1, -1), (0, -1), (1, -1),
                    (0, -2)])
    TILE_DESC = 14
    PLAY_DESC = 4
    
    OUTSIDE = np.array([3] + [0] * (TILE_DESC - 1))  # mur
    
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
        
        # contenu
        s[:, :, 0:5] = s0[:, :, 0] == np.broadcast_to(np.array([2., 3, 4, 5, 6]), (x, y, 5))
        
        # ownership (modified representation)
        s[:, :, 5] = s0[:, :, 1] + s0[:, :, 2] + s0[:, :, 3]
        
        # est un bâtiment
        s[:, :, 6] = s0[:, :, 2]
        
        # est une unité
        s[:, :, 7] = s0[:, :, 3]
        
        # type de contenu
        s[:, :, 8:14] = s0[:, :, 4] == np.broadcast_to(np.array([1., 3, 4, 5, 6, 7]), (x, y, 6))
        
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
        map_vector, player_vector = vector
        
        map_data = (map_vector.detach() + .5).int()
        player_state = (player_vector.detach() + .5).int()
        
        map_state = self.map_state.copy()
        for x, y in GameRepresentation.COO:
            if not self._is_outside(x + coo[0], y + coo[1]):
                map_state[x + coo[0], y + coo[1]] = map_data[x, y]
        
        return GameRepresentation(map_state, player_state, coo, vector)
    
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
        map_space, player_space = state_space
        
        one_action_net = {a: OneActionNet(map_space, player_space,layers) for a in action_space}
        
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
