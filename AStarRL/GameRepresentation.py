import torch
import numpy as np


class GameRepresentation:
    # COO = np.array([(0, 2),
    #                 (-1, 1), (0, 1), (1, 1),
    #                 (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
    #                 (-1, -1), (0, -1), (1, -1),
    #                 (0, -2)])
    COO = np.array([(0, 1), (-1, 0), (0, 0), (1, 0), (0, -1)])
    TILE_DESC = 14
    PLAY_DESC = 4
    
    OUTSIDE = np.array([0, 1] + [0] * (TILE_DESC - 2))  # mur
    
    DEBUG = False
    
    def __init__(self, map_state, player_state, obtained_from_coo=None, obtained_from_vector=None):
        assert type(map_state) is np.ndarray, type(map_state)
        assert type(player_state) is np.ndarray, type(player_state)
        assert np.all(map_state.astype(int) == map_state)
        assert np.all(player_state.astype(int) == player_state)
        if obtained_from_vector is not None:
            assert type(obtained_from_vector) is torch.Tensor
            assert obtained_from_vector.dtype is torch.float32
        
        self.map_state = map_state
        self.player_state = player_state
        self.obtained_from_coo = obtained_from_coo
        self.obtained_from_vector = obtained_from_vector
        
        if GameRepresentation.DEBUG:
            self.check()
    
    def check(self):
        l = list(self.map_state.reshape(-1))
        l2 = self.player_state
        if type(l2) != list:
            l2 = list(l2)
        l += l2
        # noinspection PyTypeChecker
        GameRepresentation.is_correct_vector(torch.tensor(l).unsqueeze(0), raise_=True)
    
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
        
        p = np.array([p.food, p.food_consumption, p.gold, p.lumber], dtype=float)
        assert len(p) == GameRepresentation.PLAY_DESC
        
        return GameRepresentation(s, p)
    
    @staticmethod
    def get_vector_size():
        # map + player
        return len(GameRepresentation.COO) * GameRepresentation.TILE_DESC + GameRepresentation.PLAY_DESC
    
    @staticmethod
    def get_vector_full_size(map_size):
        # map + player
        return map_size * GameRepresentation.TILE_DESC + GameRepresentation.PLAY_DESC
    
    def _is_outside(self, x, y):
        return x < 0 or y < 0 or x >= self.map_state.shape[0] or y >= self.map_state.shape[1]
    
    def _get_state_(self, x, y):
        if self._is_outside(x, y):
            return GameRepresentation.OUTSIDE
        return self.map_state[x][y]
    
    def get_player_vector(self, dtype=torch.float32):
        return torch.from_numpy(self.player_state).to(dtype=dtype)
    
    def get_vector(self, coo, dtype=torch.float32):
        map_ = [self._get_state_(x, y) for x, y in GameRepresentation.COO + coo]
        
        map_ = np.stack(map_).reshape(-1)
        
        res = np.hstack((map_, self.player_state))
        
        assert len(res) == self.get_vector_size(), (len(res), self.get_vector_size())
        return torch.tensor(res, dtype=dtype)
    
    def get_vector_full(self, dtype=torch.float32):
        res = np.hstack((self.map_state.reshape(-1), self.player_state))
        
        return torch.tensor(res, dtype=dtype)
    
    def create_from_new_vector(self, coo, vector):
        # assert vector.dtype == int
        data = (vector.detach() + .5).int()
        
        map_data = data[:-GameRepresentation.PLAY_DESC]
        map_data = map_data.reshape(len(GameRepresentation.COO), GameRepresentation.TILE_DESC)
        
        play_state = data[-GameRepresentation.PLAY_DESC:].numpy()
        # assert torch.all(play_state > 0)
        
        map_state = self.map_state.copy()
        for i, (x, y) in enumerate(GameRepresentation.COO):
            if not self._is_outside(x + coo[0], y + coo[1]):
                map_state[x + coo[0], y + coo[1]] = map_data[i]
        
        return GameRepresentation(map_state, play_state, coo, vector)
    
    @staticmethod
    def is_correct_vector(vec, raise_=False):
        if not GameRepresentation.DEBUG:
            return 
        # assert vec.shape[1] == GameRepresentation.get_vector_size()
        try:
            for vec in vec:
                vec = (vec + .5).int()
                
                map_ = vec[:-GameRepresentation.PLAY_DESC].reshape((-1, GameRepresentation.TILE_DESC))
                play = vec[-GameRepresentation.PLAY_DESC:]
                
                assert torch.all((map_ >= 0) & (map_ <= 1))
                
                # type de case
                assert torch.all(torch.sum(map_[:, 0:5], dim=1) == 1)
                
                # type d'unité/bâtiment
                assert torch.all(torch.sum(map_[:, 8:14], dim=1) <= 1)
                
                # non bâtiment ou non unité
                assert torch.all(1 - map_[:, 6] + 1 - map_[:, 7] >= 1)
                
                # 6 ou 7 => 5==1 (ou 5==2)
                assert torch.all(1 - (map_[:, 6] + map_[:, 7]) + map_[:, 5] == 1)
                
                # player
                assert torch.all(play >= 0)
                assert play[0] < 3
                assert play[1] < 3
                assert play[2] < 1000, play
                assert play[3] < 1000
        
        except AssertionError as e:
            if raise_:
                print("map")
                # noinspection PyUnboundLocalVariable
                print(tens(map_))
                for x in map_:
                    print(tens(x))
                # noinspection PyUnboundLocalVariable
                print(play)
                print("{0}".format(e))
                raise e
            else:
                return False, e
        
        return True, None
    
    def __eq__(self, other):
        return np.all(self.map_state == other.map_state) \
               and all([x == y for x, y in zip(self.player_state, other.player_state)])
    
    # def __lt__(self, other):
    #     return True
    
    def __hash__(self):
        # res = (str(self.player_state) + str(self.map_state)).__hash__()
        # res = str(self.map_state).__hash__()
        res = hash(str(self.player_state))
        # print(res)
        return res

class tens:
    def __init__(self, a):
        self.a = a
    
    def __eq__(self, other):
        # noinspection PyTypeChecker
        return torch.all(self.a == other.a)
    
    def __str__(self):
        return str(np.where(self.a == 0, " ", np.where(self.a == 1, "+", "-")))


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
