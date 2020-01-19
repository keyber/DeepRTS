import torch
from DeepRTS import Engine
from DeepRTS import python
from agent import Agent
from worldModel import GameRepresentation
import time
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # les fichiers sont créés dans le virtualEnv ...

#~/DeepRTS/virtualEnvRTS/lib/python3.7/site-packages/DeepRTS-2.5.0.dev24-py3.7-linux-x86_64.egg/DeepRTS/python/runs/


A_BUILD_TC = (13, 602, 1)
A = {
    1: {  # "n": (1, 1, -1),
        "z": (5, 11, -1),
        "q": (3, 11, -1),
        "s": (6, 11, -1),
        "d": (4, 11, -1),
        "h": (12, 4 * 6 + 30 * 11, 100),  # harverst 3+3+3+1 res <=> 4*6tics + 30 moves
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
    
    s0 = game.get_state().copy() if action[2] != -1 else None
    p.do_action(action[0])
    
    for i in range(action[1]):
        game.update()
        if i == action[2] and np.all(s0 == game.get_state()):
            # print("short circuit")
            break
    
    # for k, v in A[1].items():
    #     if v==action:
    #         print("end", k)
    #         return 


def didacticiel(game, k=3000, save_path=None):
    if save_path is not None and os.path.isfile(save_path + "_s.pth"):
        print("DIDACTICIEL LOADED")
        list_states, list_actions, list_pos = [torch.load(save_path + s + ".pth") for s in ["_s", "_a", "_p"]]
        
        assert type(list_states[0].player_state) is np.ndarray, type(list_states[0].player_state)
        print("dida nb action", len(list_actions),
              "starting state", list_states[0].player_state,
              "final state", list_states[-1].player_state)
        return list_states, list_actions, list_pos
    
    # model = Agent(s_space=GameRepresentation.get_vector_size(), action_space=A[1], objectives=None).model.nets["z"]
    
    game.reset()
    
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
    
    for i, action in enumerate(list_actions):
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
                    # print((xi, yi))
        # print(u.can_move, action, A[1][action])
        assert found == 1, found
        
        do_action(game, A[1][action])
        
        list_states.append(GameRepresentation.create_representation_from_game(game))
        
        # if action == "z" :
        # pred = model(list_states[-2].get_vector(list_pos[-1]).unsqueeze(0))
        # if torch.any(list_states[-1].get_vector(list_pos[-1]) != pred):
        #     print(list_states[-2].get_vector(list_pos[-1])[:-GameRepresentation.PLAY_DESC].reshape((len(GameRepresentation.COO), -1)))
        #     print(list_states[-1].get_vector(list_pos[-1])[:-GameRepresentation.PLAY_DESC].reshape((len(GameRepresentation.COO), -1)))
        #     print(pred[0, :-GameRepresentation.PLAY_DESC].reshape((len(GameRepresentation.COO), -1)))
        #     print("ERROR")
        #     time.sleep(10.)
        
        # if i > 0 and list_actions[i - 1] == "z" and \
        #         np.any(list_states[i].map_state != list_states[i - 1].map_state) and \
        #         (list_pos[i - 1][0] != list_pos[i][0] or list_pos[i][1] - list_pos[i - 1][1] != -1):
        #     if np.any(list_states[i].map_state != list_states[i - 1].map_state):
        #         a = list_states[i].get_vector(list_pos[i - 1]) - list_states[i - 1].get_vector(list_pos[i - 1])
        #         a = a[:-GameRepresentation.PLAY_DESC].reshape(
        #             (len(GameRepresentation.COO), GameRepresentation.TILE_DESC))
        #         print(t(a))
        #         print((list_states[i].map_state - list_states[i - 1].map_state).shape)
        # print(list_pos[i - 1][0] != list_pos[i][0])
        # print(list_pos[i][1] - list_pos[i - 1][1] != 1)
        # print(list_pos[i - 1], list_pos[i])
        # print("ERROR")
        # time.sleep(10.)
    
    print((time.time() - t0) / len(list_actions))
    
    n = len(list_pos)
    assert len(list_actions) == n, (n, len(list_actions), len(list_states))
    assert len(list_states) == n + 1, (n, len(list_pos), len(list_states))
    
    if save_path is not None:
        torch.save(list_states, save_path + "_s.pth")
        torch.save(list_actions, save_path + "_a.pth")
        torch.save(list_pos, save_path + "_p.pth")
        print("DIDACTICIEL SAVED")
    
    print("dida nb action", len(list_actions),
          "starting state", list_states[0].player_state,
          "final state", list_states[-1].player_state)
    return list_states, list_actions, list_pos


vec_max = np.array([9,9,50,300])
def rewarder(s1, s0):
    return 10 if ((s1.player_state != s0.player_state) & (s0.player_state < vec_max)).any() else -.1


def train_on_didacticiel(game, agent, agent_save_path=None, didacticiel_save_path=None):
    if agent_save_path is not None and os.path.isfile(agent_save_path + "state_model.pth"):
        agent.model.load_state_dict(torch.load(agent_save_path + "state_model.pth"))
        print("STATE MODEL LOADED")
        agent.model.eval()
        return
    
    list_states, list_actions, list_pos = didacticiel(game, save_path=didacticiel_save_path)
    
    writer = SummaryWriter()
    agent.learn(list_states, list_actions, list_pos, rewarder=rewarder, test=.3, writer=writer)
    writer.close()
    
    if agent_save_path is not None:
        torch.save(agent.model.state_dict(), agent_save_path + "state_model.pth")
        print("STATE MODEL SAVED")
    agent.model.eval()


def train_value_on_didacticiel(game, agent, agent_save_path=None, didacticiel_save_path=None):
    if agent_save_path is not None and os.path.isfile(agent_save_path + "value_model.pth"):
        agent.value_model.load_state_dict(torch.load(agent_save_path + "value_model.pth"))
        agent.reward_model.load(agent_save_path + "reward_model.pth")
        # agent.reward_model.load_state_dict(torch.load(agent_save_path + "reward_model.pth"))
        # agent.reward_model.eval()
        print("VALUE MODEL LOADED")
        agent.value_model.eval()
        return
    
    list_states, list_actions, list_pos = didacticiel(game, save_path=didacticiel_save_path)
    
    agent.train_value_model(list_states, rewarder=rewarder)
    
    if agent_save_path is not None:
        torch.save(agent.value_model.state_dict(), agent_save_path + "value_model.pth")
        # torch.save(agent.reward_model.state_dict(), agent_save_path + "reward_model.pth")
        agent.reward_model.save(agent_save_path + "reward_model.pth")
        print("VALUE MODEL SAVED")
    agent.value_model.eval()
    # agent.reward_model.eval()


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
    engine_config.set_start_lumber(540)  # Lumber de départ
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
    
    def is_goal(s, t):
        for ss, tt in zip(s.player_state, t):
            if tt != -1 and ss < tt:
                return False
        return True
    
    # s_space = ((len(GameRepresentation.COO), GameRepresentation.TILE_DESC), GameRepresentation.PLAY_DESC)
    game_size = GameRepresentation.get_vector_full_size(game.get_height() * game.get_height())
    agent = Agent(GameRepresentation.get_vector_size(), game_size, action_space=A[1], objectives=(targets[0], is_goal))
    
    train_on_didacticiel(game, agent, agent_save_path="/home/keyvan/DeepRTS/AStarRL/",
                         didacticiel_save_path="/home/keyvan/DeepRTS/AStarRL/dida")
    train_value_on_didacticiel(game, agent, agent_save_path="/home/keyvan/DeepRTS/AStarRL/",
                               didacticiel_save_path="/home/keyvan/DeepRTS/AStarRL/dida")
    game.reset()
    do_action(game, A_BUILD_TC)
    game.players[0].do_manual_action(LEFT_CLICK, 3, 8)
    # for a in "zdzdddzdddd":
    #     do_action(game, A[1][a])
    #     print(agent.value_model(GameRepresentation.create_representation_from_game(game).get_vector_full()).item())
    # exit()
    agent.reset(game)
    print("début de la partie")

    while not game.is_terminal():
        t0 = time.time()
        pos, a = agent.act()
        print((pos[0], pos[1]), a, A[1][a], "%.1f" % (time.time() - t0))
        do_action(game, A[1][a], pos)
        
        agent.get_result(game, reward=0)


# a = np.arange(0,60).reshape((5,12))
# print(a[(0,0), (3,4), (4,3)])
main()
