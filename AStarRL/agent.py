import torch
from torch import nn
from torch.optim import Adam
from worldModel import GameRepresentation, WorldModel
import numpy as np
import heapq
import matplotlib.pyplot as plt

a_2_iOut = {"z": 2, "q": 5, "s": 10, "d": 7}

def int_loss(pred, true, trg_idx, src_idx=6):
    batch_size = pred.shape[0]
    
    pred = (pred.detach()+.5).int()
    diff = (true - pred).abs()
    
    m, p = diff[:, :-GameRepresentation.PLAY_DESC], diff[:, -GameRepresentation.PLAY_DESC:]
    m = torch.sum(m.reshape((batch_size, len(GameRepresentation.COO), GameRepresentation.TILE_DESC)), dim=2)
    
    return {
        "map_n_wrongs": torch.sum(m != 0, dim=1, dtype=torch.float32).mean(),
        "player_n_wrongs": torch.sum(p != 0, dim=1, dtype=torch.float32).mean(),
        "map_src_count":m[:, src_idx].mean(),
        "map_trg_count":m[:, trg_idx].mean(),
    }
    
class RepresLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_loss = nn.BCEWithLogitsLoss()
        self.player_loss = nn.MSELoss()
    
    def forward(self, pred, true):
        pred_map, pred_play = pred
        true_map, true_play = true
        return 
class Agent:
    def __init__(self, s_space, action_space, objectives):
        self.epoch = 0
        self.t = 0
        self.learning_period = 10
        self.action_space = action_space
        
        self.model = WorldModel(s_space, action_space, layers=[512, 256])  #type: WorldModel
        self.model_loss = nn.SmoothL1Loss()
        self.model_optim = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e0)
        self.model_max_loss = 1e-3
        self.warmup_loss_convergence_eps = 1e-3
        self.warmup_max_iter = 100
        
        self.mem_state0 = []
        self.mem_action = []
        self.mem_state1 = []
        
        self.last_state = None
        self.last_action = None
        self.predicted_state = None
        self.objectives = objectives
    
    def learn(self, list_states, list_actions, list_pos, test=None, writer=None):
        if test is None:
            test = {}
        
        action_transitions = {}
        for i, action in enumerate(list_actions):
            if action not in action_transitions:
                action_transitions[action] = []
            action_transitions[action].append((list_states[i], list_pos[i], list_states[i + 1]))
        
        plt.ion()
        ll = {action: [] for action in action_transitions}
        ll_test = {action: [] for action in action_transitions}
        
        for action, transitions in action_transitions.items():
            print("learning action", action)
            x = torch.stack([s0.get_vector(pos) for s0, pos, _s in transitions])
            y_true = torch.stack([s1.get_vector(pos) for _s, pos, s1 in transitions])
            
            loss_old = [self.warmup_loss_convergence_eps * i for i in range(5)]
            
            for epoch in range(self.warmup_max_iter):
                if loss_old[-1] < self.model_max_loss:
                    diff = sum(abs(loss_old[i] - loss_old[i + 1]) for i in range(len(loss_old) - 1))
                    if diff < self.warmup_loss_convergence_eps:
                        break
                
                y_pred = self.model.forward_vec(action, v_in=x)
                l = self.model_loss(y_pred_map, y_true)
                
                self.model_optim.zero_grad()
                l.backward()
                self.model_optim.step()
                
                loss_old.pop(0)
                loss_old.append(l.item())
                
                if action in test:
                    y_pred_test = self.model.forward_vec(action, v_in=test[action][0])
                    # if int_loss:
                    #     l_test = (test[action][1]-(y_pred_test.detach()+.5).int()).abs().mean()
                    # else:
                    l_test = self.model_loss(y_pred_test, test[action][1])
                    ll_test[action].append(l_test.item())
                
                ll[action].append(l.item())
                
                if writer is not None:
                    losses = int_loss(y_pred, y_true, trg_idx=a_2_iOut[action])
                    losses["mse"] = l.item()
                    writer.add_scalars("loss", losses, epoch)
                
                # print(ll[action][-1])
                for a in ll:
                    plt.plot(range(len(ll[a])), np.log10(ll[a]), label=a)
                for a in ll_test:
                    plt.plot(range(len(ll_test[a])), np.log10(ll_test[a]), label=a + "t")
                
                plt.legend()
                plt.draw()
                plt.pause(1e-6)
                plt.clf()
            
            else:
                print("warning model did not finish to learn action", action)
            print(i, "iteration", "last loss train %.2f %.2f" % (loss_old[-2], loss_old[-1]))
            if action in test:
                print("last loss test %.2f" % l_test)
            writer.close()
            exit()
    
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
