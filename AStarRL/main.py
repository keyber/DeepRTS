import torch
from DeepRTS import Engine
from DeepRTS import python
from agent import Agent
from worldModel import GameRepresentation
import time
import os
import random
from torch.utils.tensorboard import SummaryWriter # les fichiers sont créés dans le virtualEnv ...


A_BUILD_TC = (13, 601, 1)
A = {
    1: {  # "n": (1, 1, -1),
        "z": (5, 11, -1),
        "q": (3, 11, -1),
        "s": (6, 11, -1),
        "d": (4, 11, -1),
        "h": (12, 301, 6),
        # "0": (13, 601, 1),
        # "1": (14, 10, -1),
        # "2": (15, 10, -1),
        # "w": (16, 100, -1),
    },
    3: {
        "0": (13, 101, -1)
    }
}

possible_states = [[2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6],  #herbe, mur, ...
                   [0, 1, 2],
                   [0, 1],
                   [0, 1],
                   [0, 1],  #, 3, 4, 5, 6, 7],  # peon, TC, ...
                   [.5],
                   [8],  # spawn, walk, ...
                   [0],
                   [0],
                   [7],
                   [0]]

possible_states_p = [[0, 1],
                     [0, 1, 2],
                     [0, 249, 250, 499, 500],
                     [0, 249, 250, 499, 500], ]

LEFT_CLICK = 1


def _main_do_action(A, game, list_action, reset=True):
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
                if s[xi][yi][5] == 1 and s[xi][yi][7] == 1:
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
    
    s0 = GameRepresentation.create_representation_from_game(game) if action[2] != -1 else None
    p.do_action(action[0])
    game.update()
    
    for i in range(action[1]):
        game.update()
        if i == action[2] and s0 == GameRepresentation.create_representation_from_game(game):
            # short circuit
            break


def didacticiel(game, k, save_path=None):
    if save_path is not None and os.path.isfile(save_path + "_s.pth"):
        print("DIDACTICIEL LOADED")
        list_states, list_actions, list_pos = [torch.load(save_path + s + ".pth") for s in ["_s", "_a", "_p"]]
        return list_states, list_actions, list_pos
    
    t0 = time.time()
    p = game.players[0]
    list_states = []
    # construit un TC, fait le tour, va en haut à droite, récolte
    list_actions = "zddszzzdsddsdsdssqsqqzqqzzdzzdzdzdzdzdzhhsdsdhsdsdsdsdsdsdh"
    for a in random.choices(list(A[1].keys()), k=k):
        list_actions += 3 * a
    
    do_action(game, A_BUILD_TC)
    
    list_states.append(GameRepresentation.create_representation_from_game(game))
    
    list_pos = []
    
    for action in list_actions:
        # print(p.get_targeted_unit(), action, A[action], p.food, p.food_consumption, p.gold, p.lumber)
        s = list_states[-1].map_state
        # for xi in range(s.shape[0]):
        #     for yi in range(s.shape[1]):
        #         if s[xi][yi][5] == 1:
        #             p.do_manual_action(LEFT_CLICK, xi, yi)
        #             game.update()
        #             u = p.get_targeted_unit()
        #             print(xi, yi, u.can_move)
        found = 0
        for xi in range(s.shape[0]):
            for yi in range(s.shape[1]):
                if s[xi][yi][5] == 1 and s[xi][yi][7] == 1:
                    p.do_manual_action(LEFT_CLICK, xi, yi)
                    game.update()
                    # u = p.get_targeted_unit()
                    # print(u)
                    found += 1
                    list_pos.append((xi, yi))
        # print(u.can_move, action, A[1][action])
        assert found == 1, found
        # time.sleep(.1)
        
        do_action(game, A[1][action])
        
        list_states.append(GameRepresentation.create_representation_from_game(game))
    
    print((time.time() - t0) / len(list_actions))
    
    n = len(list_pos)
    assert len(list_actions) == n, (n, len(list_actions), len(list_states))
    assert len(list_states) == n + 1, (n, len(list_pos), len(list_states))
    
    if save_path is not None:
        torch.save(list_states, save_path + "_s.pth")
        torch.save(list_actions, save_path + "_a.pth")
        torch.save(list_pos, save_path + "_p.pth")
        print("DIDACTICIEL SAVED")
    
    return list_states, list_actions, list_pos


def gen_dataset(size):
    assert 0, "indices à revoir" #todo
    test = {}
    a_2_iOut = {"z": 2, "q": 5, "s": 10, "d": 7}
    
    for a in "zqsd":
        test_x = []
        test_y = []
        for _ in range(size):
            i_in = 6
            i_out = a_2_iOut[a]
            
            v = [[random.choice(possible_states[i]) for i in range(GameRepresentation.TILE_DESC)]
                 for _ in range(len(GameRepresentation.COO))]
            
            for c in v:
                if c[0] != 2:
                    for i in range(1, GameRepresentation.TILE_DESC):
                        c[i] = 0
                if c[2] == 1 or c[3] == 1:
                    c[1] = random.randint(1, 2)
                if c[2] == 1:
                    c[4] = random.choice([3, 4])
                if c[3] == 1:
                    c[4] = random.choice([1, 5, 7])
            
            v[i_in] = [2, 1, 0, 1, 1, .5, 8, 0, 0, 7, 0]
            
            p = [random.choice(possible_states_p[i]) for i in range(GameRepresentation.PLAY_DESC)]
            
            v_in = sum(v, [])
            v_in += p
            test_x.append(torch.tensor(v_in, dtype=torch.float32))
            
            t = v[i_out]
            s = v[i_in]
            
            # déplacement si case libre et on est une unité 
            if t[0] == 2 and t[1] == 0 and s[7] == 1:
                v[i_in], v[i_out] = v[i_out], v[i_in]
            
            # p ne change pas
            test_y.append(torch.tensor(sum(v, []) + p, dtype=torch.float32))
        
        test[a] = (torch.stack(test_x), torch.stack(test_y))
    
    return test


def train_on_didacticiel(game, agent, agent_save_path=None, didacticiel_save_path=None,
                         didacticiel_size=1000, test_size=0):
    if agent_save_path is not None and os.path.isfile(agent_save_path):
        print("MODEL LOADED")
        agent.model.load_state_dict(torch.load(agent_save_path))
        return
    
    test = gen_dataset(test_size) if test_size else None
    
    game.reset()
    list_states, list_actions, list_pos = didacticiel(game, k=didacticiel_size, save_path=didacticiel_save_path)
    
    writer = SummaryWriter()
    agent.learn(list_states, list_actions, list_pos, test=test, writer=writer)
    writer.close()
    
    if agent_save_path is not None:
        print("MODEL SAVED")
        torch.save(agent.model.state_dict(), agent_save_path)


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
    engine_config.set_start_gold(900)  # Or de départ
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
    
    game.reset()
    time.sleep(1.)
    
    # _main_do_action(game, list_action=A)
    # _main_do_action(game, list_action=["d"] * 1000)
    # _main_do_action(game, list_action="0zdzdzdzdzdzdzdhzdzdzd", reset=False)
    
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
    
    s_space = ((len(GameRepresentation.COO), GameRepresentation.TILE_DESC), GameRepresentation.PLAY_DESC)
    agent = Agent(s_space=s_space, action_space=A[1], objectives=(targets[0], H, is_goal))
    
    train_on_didacticiel(game, agent, agent_save_path="/home/keyvan/DeepRTS/AStarRL/world_model0.pth",
                         didacticiel_save_path="/home/keyvan/DeepRTS/AStarRL/dida")
    
    game.reset()
    
    do_action(game, A_BUILD_TC)
    
    agent.reset(game)
    print("début de la partie")
    while not game.is_terminal():
        pos, a = agent.act()
        print((pos[0], pos[1]), a, A[1][a])
        do_action(game, A[1][a], pos)
        
        agent.get_result(game, _reward=0)


# a = np.arange(0,60).reshape((5,12))
# print(a[(0,0), (3,4), (4,3)])
main()
