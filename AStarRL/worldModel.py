import torch
from torch import nn
from torch.optim import Adam
from DeepRTS import Engine
from DeepRTS import python
import time
import numpy as np
import heapq
import os
import matplotlib.pyplot as plt
import random

A = {
    # "n": (1, 1, -1),
    "z": (5, 11, -1),
    "q": (3, 11, -1),
    "s": (6, 11, -1),
    "d": (4, 11, -1),
    "h": (12, 301, 10),
    "0": (13, 601, 1),
    # "1": (14, 10, -1),
    # "2": (15, 10, -1),
    # "w": (16, 100, -1),
}

LEFT_CLICK = 1


class OneActionNet(nn.Module):
    def __init__(self, state_space, layers):
        super().__init__()
        assert layers[-1] == state_space
        
        l = []
        size = state_space
        for x in layers:
            l.append(nn.Linear(size, x))
            l.append(nn.ReLU())
            size = x
        l.append(nn.Linear(size, state_space))
        
        self.layers = nn.Sequential(*l)
    
    def forward(self, x):
        return self.layers(x)


class GameRepresentation:
    COO = np.array([(0, 2),
                    (-1, 1), (0, 1), (1, 1),
                    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
                    (-1, -1), (0, -1), (1, -1),
                    (0, -2)])
    TILE_DESC = 11
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
        s = np.empty((game.get_width(), game.get_height(), 11), dtype=int)
        s[:, :, :7] = game.get_state()[:, :, :7]
        s[:, :, [7, 8]] = np.zeros((game.get_width(), game.get_height(), 2))
        s[:, :, [9, 10]] = game.get_state()[:, :, [8, 9]]
        
        # modify ownership representation
        s[:, :, 1] = s[:, :, 1] + s[:, :, 2] + s[:, :, 3]
        
        for xi in range(s.shape[0]):
            for yi in range(s.shape[1]):
                if s[xi][yi][1] == 1:
                    p.do_manual_action(LEFT_CLICK, xi, yi)
                    u = p.get_targeted_unit()
                    assert u is not None
                    
                    s[yi, xi, [7, 8]] = [u.gold_carry, u.lumber_carry]
        
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
        local_state_length = len(GameRepresentation.COO) * GameRepresentation.TILE_DESC
        
        data = (vector.detach() + .5).int()[:local_state_length]
        data = data.reshape((len(GameRepresentation.COO), GameRepresentation.TILE_DESC))
        
        map_state = self.map_state.copy()
        for x, y in GameRepresentation.COO:
            if not self._is_outside(x + coo[0], y + coo[1]):
                map_state[x + coo[0], y + coo[1]] = data[x, y]
        
        player_state = vector.detach()[local_state_length:]
        
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
        
        one_action_net = {}
        
        for action in action_space:
            l = []
            size = state_space
            for x in layers:
                l.append(nn.Linear(size, x))
                l.append(nn.ReLU())
                size = x
            l.append(nn.Linear(size, state_space))
            
            one_action_net[action] = nn.Sequential(*l)
        
        self.nets = nn.ModuleDict(one_action_net)
    
    def forward_full(self, game_repres, pos, action):
        v_out = self.forward_vec(action, game_repres, pos)
        
        return game_repres.create_from_new_vector(pos, v_out)
    
    def forward_vec(self, action, game_repres=None, pos=None, v_in=None):
        if v_in is None:
            v_in = game_repres.get_vector(pos)
        
        # noinspection PyUnresolvedReferences
        return self.nets[action](v_in)


class Agent:
    def __init__(self, s_space, action_space, objectives):
        self.epoch = 0
        self.t = 0
        self.learning_period = 10
        self.action_space = action_space
        
        self.model = WorldModel(s_space, action_space, layers=[s_space, s_space])  #type: WorldModel
        self.model_loss = nn.MSELoss()
        self.model_optim = Adam(self.model.parameters(), lr=1e-3)
        self.model_max_loss = .5
        self.warmup_loss_convergence_eps = 1e-1
        self.warmup_max_iter = 1000
        
        self.mem_state0 = []
        self.mem_action = []
        self.mem_state1 = []
        
        self.last_state = None
        self.last_action = None
        self.predicted_state = None
        self.objectives = objectives
    
    def learn(self, list_states, list_actions, list_pos):
        action_transitions = {}
        for i, action in enumerate(list_actions):
            if action not in action_transitions:
                action_transitions[action] = []
            action_transitions[action].append((list_states[i], list_pos[i], list_states[i + 1]))
        
        plt.ion()
        ll = {action: [] for action in action_transitions}
        for action, transitions in action_transitions.items():
            print("learning action", action)
            x = torch.stack([s0.get_vector(pos) for s0, pos, _s in transitions])
            y_true = torch.stack([s1.get_vector(pos) for _s, pos, s1 in transitions])
            
            loss_old = [self.warmup_loss_convergence_eps * i for i in range(5)]
            
            for i in range(self.warmup_max_iter):
                if loss_old[-1] < self.model_max_loss:
                    diff = sum(abs(loss_old[i] - loss_old[i + 1]) for i in range(len(loss_old) - 1)) / len(loss_old)
                    if diff < self.warmup_loss_convergence_eps:
                        break
                
                y_pred = self.model.forward_vec(action, v_in=x)
                l = self.model_loss(y_pred, y_true)
                
                self.model_optim.zero_grad()
                l.backward()
                self.model_optim.step()
                
                loss_old.pop(0)
                loss_old.append(l.item())
                
                print(l.item())
                ll[action].append(l.item())
                for a in ll:
                    plt.plot(range(len(ll[a])), ll[a], label=a)
                plt.legend()
                plt.draw()
                plt.pause(1e-6)
                plt.clf()
            
            if loss_old[-1] >= self.model_max_loss:
                print("warning model did not finish to learn action", action, )
    
    def update(self):
        self.t += 1
        
        if not (self.t % self.learning_period or self.t):
            return
        
        # regarde la dernière transition
        predicted_states = self.model(self.mem_state0[-1], self.mem_action[-1])
        true_states = self.mem_state1[-1]
        
        l = self.model_loss(predicted_states, true_states)
        
        # si on se trompe, revoit tous les exemples
        if l > self.model_max_loss:
            print("revoit tout")
            self.model_optim.zero_grad()
            for s0, a, s1 in zip(self.mem_state0[:-1], self.mem_action[:-1], self.mem_state1[:-1]):
                l += self.model_loss(self.model(s0, a), s1)
            l.backward()
            self.model_optim.step()
    
    def act(self):
        n = a_star(self.model, self.action_space, self.last_state, *self.objectives)
        assert n[3] is not None, n[3]
        while n[3][3] is not None:
            n = n[3]
        
        assert n[4] is not None
        self.last_action = n[4]
        self.predicted_state = n[2]
        return self.last_action
    
    def get_result(self, game, _reward):
        # self.update()
        new_state = GameRepresentation.create_representation_from_game(game)
        
        y_pred = self.predicted_state.obtained_from_vector
        y_true = new_state.get_vector(self.predicted_state.obtained_from_coo)
        l = self.model_loss(y_pred, y_true)
        if l > 1e-3:
            print("prediction error, loss:", l.item())
            # print("pred", y_pred)
            # print("true", y_true)
            # print()
            self.model_optim.zero_grad()
            l.backward()
            self.model_optim.step()
        else:
            print("well predicted")
        
        self.last_state = new_state
    
    def reset(self, game):
        self.last_state = GameRepresentation.create_representation_from_game(game)
        self.last_action = None


def a_star(model, action_space, s0, t, H, is_goal, max_iter=1):
    h = H(s0, t)
    # estimation total, dont heuristique, state, noeud précédent, action_précédente
    opened_states = [(0 + h, h, s0, None, None)]
    closed_states = set()
    
    best_node = list(opened_states[0])
    
    for i in range(max_iter + 1):
        if not len(opened_states):
            break
        
        node = opened_states.pop()
        closed_states.add(node)
        _e, _h, s, _prev_node, _prev_a = node
        d = _e - _h
        
        if is_goal(s, t):
            print("goal found", i, _e, _h, s.player_state)
            return node
        
        if _e < best_node[0] or best_node[3] is None:
            best_node = node
        
        if i == max_iter:
            break
        
        for xi in range(s.map_state.shape[0]):
            for yi in range(s.map_state.shape[1]):
                if s.map_state[xi][yi][1] == 1:
                    # print("A* :", xi, yi, len(opened_states))
                    for a in action_space:
                        s2 = model.forward_full(s, (xi, yi), a)
                        
                        if s2 in closed_states: continue
                        
                        h2 = H(s2, t)
                        d2 = d + 1
                        
                        heapq.heappush(opened_states, (d2 + h2, h2, s2, node, ((xi, yi), a)))
                    # print("A* :", xi, yi, len(opened_states))
    
    assert best_node is not None
    return best_node


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


def _main_do_action(game, list_action, reset=True):
    game.reset()
    time.sleep(1.)
    
    p = game.players[0]
    # pprint(GameRepresentation.create_representation_from_game(game), p)
    for action in list_action:
        if reset:
            game.reset()
        
        s0 = GameRepresentation.create_representation_from_game(game)
        
        s = s0.map_state
        found = False
        for xi in range(s.shape[0]):
            for yi in range(s.shape[1]):
                if s[xi][yi][1] == 1 and s[xi][yi][4] == 1:
                    found = True
                    p.do_manual_action(LEFT_CLICK, xi, yi)
        assert found
        
        p.do_action(A[action][0])
        found = 0
        for i in range(A[action][1]):
            game.update()
            s1 = GameRepresentation.create_representation_from_game(game)
            # pprint(s1, p)
            if s1 != s0:
                s0 = s1
                print("n steps for macro", action, ":", i)
                found += 1
                # pprint(s0, p)
        
        if not found:
            print("no change for macro", action, "in", A[action][1], "steps")
    
    exit()


def do_action(game, action, pos=None):
    p = game.players[0]
    if pos is not None:
        p.do_manual_action(LEFT_CLICK, pos[0], pos[1])
        game.update()
    
    s0 = GameRepresentation.create_representation_from_game(game) if A[action][2] != -1 else None
    p.do_action(A[action][0])
    game.update()
    
    for i in range(A[action][1] + 1):
        game.update()
        if i == A[action][2] and s0 == GameRepresentation.create_representation_from_game(game):
            # short circuit
            break
    time.sleep(.1)


def didacticiel(game, k=10):
    p = game.players[0]
    list_states = []
    # construit un TC, fait le tour, va en haut à droite, récolte
    list_actions = "zdds0zzz0dsdds0dsdssqsqqzq0qzzdzzdzdzdz0dzdzhhsdsd0hsdsdsdsdsdsd0h"
    list_actions += "".join(random.choices(list(A.keys()), k=k))
    list_states.append(GameRepresentation.create_representation_from_game(game))
    
    list_pos = []
    
    for action in list_actions:
        print(p.get_targeted_unit(), action, A[action], p.food, p.food_consumption, p.gold, p.lumber)
        s = list_states[-1].map_state
        found = False
        for xi in range(s.shape[0]):
            for yi in range(s.shape[1]):
                if s[xi][yi][1] == 1 and s[xi][yi][4] == 1:
                    found = True
                    p.do_manual_action(LEFT_CLICK, xi, yi)
                    game.update()
                    u = p.get_targeted_unit()
                    print(xi, yi, u, u.lumber_carry, u.gold_carry)
                    list_pos.append((xi, yi))
        assert found
        
        do_action(game, action)
        
        list_states.append(GameRepresentation.create_representation_from_game(game))
        
        print()
    
    n = len(list_pos)
    assert len(list_actions) == n, (n, len(list_actions), len(list_states))
    assert len(list_states) == n + 1, (n, len(list_pos), len(list_states))
    
    return list_states, list_actions, list_pos


def train_on_didacticiel(game, agent, save_path=None):
    if save_path is not None and os.path.isfile(save_path):
        print("MODEL LOADED")
        agent.model.load_state_dict(torch.load(save_path))
        return
    
    game.reset()
    list_states, list_actions, list_pos = didacticiel(game)
    agent.learn(list_states, list_actions, list_pos)
    
    if save_path is not None:
        print("MODEL SAVED")
        torch.save(agent.model.state_dict(), save_path)


def main():
    engine_config = Engine.Config()  # Création de la configuration
    engine_config.set_archer(True)  # Autoriser les archers
    engine_config.set_barracks(True)  # Autoriser les baraquement
    engine_config.set_farm(True)  # Autoriser les fermes
    engine_config.set_footman(True)  # Autoriser l’infanterie
    engine_config.set_auto_attack(False)  #Attaquer automatiquement si on est attaqué
    engine_config.set_food_limit(1000)  # Pas plus de 1000 unités
    engine_config.set_harvest_forever(False)  # Récolter automatiquement
    engine_config.set_instant_building(False)  # Temps de latence ou non pour la construction
    engine_config.set_pomdp(False)  # Pas de brouillard (partie de la carte non visible)
    engine_config.set_console_caption_enabled(False)  # ne pas afficher des infos dans la console
    engine_config.set_start_lumber(500)  # Lumber de départ
    engine_config.set_start_gold(500)  # Or de départ
    engine_config.set_instant_town_hall(False)  # Temps de latence ou non pour la construction d’un townhall
    engine_config.set_terminal_signal(True)  # Connaître la fin du jeu
    
    gui_config = python.Config(render=True,  #activer la GUI
                               view=True,
                               inputs=False,  #interagir avec un joueur humain
                               caption=True,
                               unit_health=True,
                               unit_outline=False,
                               unit_animation=True,
                               audio=False)
    
    MAP = python.Config.Map.TWENTYONE
    
    game = python.Game(MAP, n_players=1, engine_config=engine_config, gui_config=gui_config)
    game.set_max_fps(int(1e9))  #augmenter les fps lorsqu’on ne veut pas visualiser le jeu
    game.set_max_ups(int(1e9))
    
    # _main_do_action(game, list_action=A)
    # _main_do_action(game, list_action=["d"] * 1000)
    _main_do_action(game, list_action="0zdzdzdzdzdzdzdhzdzdzd", reset=False)
    
    targets = [
        # [2, -1, -1, -1],   # créer un TC
        [-1, -1, -1, 500],  # récolter du bois
        # [-1, 2, -1, -1],   # créer un péon
    ]
    
    def H(s, t):
        s = s.player_state
        res = torch.zeros(1, dtype=torch.float32)
        
        for ss, tt in zip(s, t):
            if tt != -1 and ss < tt:
                res += tt - ss
        
        res -= sum(s) * .0001
        return res
    
    def is_goal(s, t):
        for ss, tt in zip(s.player_state, t):
            if tt != -1 and ss < tt:
                return False
        return True
    
    agent = Agent(s_space=GameRepresentation.get_vector_size(), action_space=A, objectives=(targets[0], H, is_goal))
    
    train_on_didacticiel(game, agent, save_path="/home/keyvan/DeepRTS/AStarRL/world_model0.pth")
    
    game.reset()
    
    do_action(game, "0")
    
    agent.reset(game)
    print("début de la partie")
    while not game.is_terminal():
        pos, a = agent.act()
        print((pos[0], pos[1]), a, A[a])
        do_action(game, a, pos)
        
        agent.get_result(game, _reward=0)


# a = np.arange(0,60).reshape((5,12))
# print(a[(0,0), (3,4), (4,3)])
main()
