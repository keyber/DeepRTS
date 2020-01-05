from DeepRTS import Engine
from DeepRTS import python
import time
import numpy as np

A = {
    "z": (5, 10),
    "q": (3, 10),
    "s": (6, 10),
    "d": (4, 10),
    "h": (12, 300),
    "0": (13, 595),
    "1": (14, 10),
    "2": (15, 10),
    "w": (16, 100),
}

LEFT_CLICK = 1


class GameRepresentation:
    def __init__(self, map_state, player_state):
        self.map_state = map_state
        self.player_state = player_state
    
    @staticmethod
    def create_representation_from_game(game):
        p = game.players[0]
        s = np.empty((game.get_width(), game.get_height(), 11))
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
        
        return GameRepresentation(s, p)


def didacticiel(game):
    p = game.players[0]
    
    # construit un TC, fait le tour, monte en haut à droite, récolte, va en bas à droite récolte
    list_actions = "zdds0zzzdsddsdsdssqsqqzqqzzdzzdzdzdzdzdzhhhsdsdsdsdsdsdsdsdsdshhzzzzddddssssqqq"
    
    list_states = []
    list_states.append(GameRepresentation.create_representation_from_game(game))
    
    print(*[x.rjust(10) for x in "xi, yi, action, code,wait, food, conso, p.gold, p.lumber, u.gold, u.lumber".split(", ")])
    
    for action in list_actions:
        # affiche les unités de la map
        s = list_states[-1].map_state
        for xi in range(s.shape[0]):
            for yi in range(s.shape[1]):
                if s[xi][yi][1] == 1 and s[xi][yi][4] == 1:
                    p.do_manual_action(LEFT_CLICK, xi, yi)
                    game.update()
                    u = p.get_targeted_unit()
                    print(*[x.rjust(10) for x in map(str, [xi, yi, action, A[action],
                          p.food, p.food_consumption, p.gold, p.lumber,
                          u.lumber_carry, u.gold_carry])])
                    
        
        # effectue l'action
        # update et sleep plein de fois pour que l'action se réalise            
        p.do_action(A[action][0])
        game.update()
        
        time.sleep(.1)
        for i in range(A[action][1] + 1):
            game.update()
            time.sleep(.01)
        time.sleep(.1)
        
        list_states.append(GameRepresentation.create_representation_from_game(game))
    
    return list_states, list_actions


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
    game.set_max_fps(int(1e6))  #augmenter les fps lorsqu’on ne veut pas visualiser le jeu
    game.set_max_ups(int(1e6))
    
    game.reset()
    didacticiel(game)


main()
