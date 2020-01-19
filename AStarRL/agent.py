import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from worldModel import GameRepresentation, WorldModel, tens
from valueModel import ValueModel2, RewardModel2
import numpy as np
import matplotlib.pyplot as plt
import bfs

a_2_iOut = {"z": 2, "q": 5, "s": 10, "d": 7}


def int_loss(x, pred, true, trg_idx=None, src_idx=0, verbose=0):
    pred = pred.detach()
    true = true.detach()
    pred_map, pred_play = pred[:, :-GameRepresentation.PLAY_DESC], pred[:, -GameRepresentation.PLAY_DESC:]
    true_map, true_play = true[:, :-GameRepresentation.PLAY_DESC], true[:, -GameRepresentation.PLAY_DESC:]
    
    pred_map_int = (pred_map + .5).int()
    pred_play_int = (pred_play + .5).int()
    
    m = (true_map - pred_map_int).abs()  # batch, map_size, tile_size
    p = (true_play - pred_play_int).abs()  # batch, player_size
    
    assert torch.all(m == m.int().float())
    assert torch.all(p == p.int().float())
    
    m = m.reshape((m.shape[0], len(GameRepresentation.COO), GameRepresentation.TILE_DESC))
    
    if verbose:
        dbg_transitions = []
        if len(dbg_transitions) == 0:
            x2 = x[:, :-GameRepresentation.PLAY_DESC].reshape_as(m)
            diff = true_map.reshape_as(m) - x2
            
            cpt = [0] * 2
            for i in range(len(m)):
                d = tens(diff[i])
                if d not in dbg_transitions:
                    dbg_transitions.append(d)
                    print("ajout de la transition")
                    print(d)
                    print("obtenu à partir de l'état")
                    print(tens(x2[i]))
                    print()
                cpt[dbg_transitions.index(d)] += 1
            
            print(cpt)
        
        assert len(dbg_transitions) == 2
        cpt = [0, 0]
        
        x2 = x[:, :-GameRepresentation.PLAY_DESC].reshape_as(m)
        diff = true_map.reshape_as(m) - x2
        for i in range(len(m)):
            d = tens(diff[i])
            if torch.any(m[i] != 0):
                cpt[dbg_transitions.index(d)] += 1
                # print("transition", dbg_transitions.index(d))
                # print(x2[i])
                # print()
                
                # for j, l in enumerate(x2[i]):
                #     print(j, "-" if torch.all(l==0) else l)
                # print()
                # for j, l in enumerate(m[i]):
                #     print(j, "-" if torch.all(l==0) else l)
                # print()
                # for j, l in enumerate(true_map.reshape_as(m)[i]):
                #     print(j, "-" if torch.all(l==0) else l)
                # print()
                # for j, l in enumerate((true_map.reshape_as(m) - x)[i]):
                #     print(j, "-" if torch.all(l==0) else l)
                # for j, l in enumerate((true_map - pred_map_int).reshape_as(m)[i]):
                #     print(j, "-" if torch.all(l==0) else l)
                # for j, l in enumerate(pred_map_int.reshape_as(m)[i]):
                #     print(j, "-" if torch.all(l==0) else l)
                
                # print()
                # input_ = x[i].clone()
                # input_[-GameRepresentation.PLAY_DESC:]=0
                # input_ = torch.zeros_like(input_)
                # input_[4*14+0]=1
                # print(torch.matmul(net.layers[-1].weight, input_)[:-GameRepresentation.PLAY_DESC].reshape_as(m[i]))
                
                # print("\n")
        
        # print()
        print(cpt, sum(cpt))
    m = torch.sum(m, dim=2)
    
    res = {
        "map_n_wrongs": torch.sum(m != 0, dim=1, dtype=torch.float32).mean(),
        "player_n_wrongs": torch.sum(p != 0, dim=1, dtype=torch.float32).mean(),
        "map_src_count": m[:, src_idx].mean(),
        "map_mse": nn.L1Loss()(pred_map, true_map)
    }
    if trg_idx is not None:
        res["map_trg_count"] = m[:, trg_idx].mean()
    
    return res


class RepresLoss(nn.Module):
    def __init__(self, player_weight=1.):
        super().__init__()
        self.map_loss = nn.BCELoss(reduction="none")
        self.mse = nn.MSELoss()
        self.player_weight = player_weight
    
    def forward(self, pred, true):
        pred_map, pred_play = pred[:, :-GameRepresentation.PLAY_DESC], pred[:, -GameRepresentation.PLAY_DESC:]
        true_map, true_play = true[:, :-GameRepresentation.PLAY_DESC], true[:, -GameRepresentation.PLAY_DESC:]
        
        # pred_map_int = (pred_map+.5).int()
        # m = (true_map - pred_map_int).abs()
        # bce = nn.BCELoss(weight=torch.where(m!=0, torch.tensor([1.]), torch.tensor([0.])))
        
        # map_loss = bce(pred_map, true_map)
        # map_loss = nn.BCEWithLogitsLoss()(pred_map, true_map)
        # map_loss = torch.mean(self.map_loss(pred_map, true_map))
        # map_loss = torch.mean(torch.pow(self.map_loss(pred_map, true_map), 2))
        map_loss = self.mse(pred_map, true_map)
        # map_loss = torch.zeros(1)
        
        play_loss = self.mse(pred_play, true_play)
        # play_loss = torch.zeros(1)
        
        return map_loss + self.player_weight * play_loss, (map_loss, play_loss)


class Agent:
    def __init__(self, world_model_state_space, value_model_state_space, action_space, objectives):
        self.epoch = 0
        self.t = 0
        self.learning_period = 10
        self.action_space = action_space
        
        self.model = WorldModel(world_model_state_space, action_space, layers=[])  #type: WorldModel
        self.model_loss = RepresLoss()
        self.model_optim = Adam(self.model.parameters(), lr=1e-1)  #, weight_decay=0)
        self.model_sched = ExponentialLR(self.model_optim, gamma=.1)
        self.model_max_loss = 1e-4
        self.warmup_loss_convergence_eps = 1e-4
        self.warmup_max_iter = 100
        
        self.mem_state0 = []
        self.mem_action = []
        self.mem_state1 = []
        
        self.value_model = ValueModel2(value_model_state_space)
        # self.value_model = ValueModel(value_model_state_space, action_space)
        # self.value_model = nn.Linear(value_model_state_space, 1)
        self.value_model_loss = nn.MSELoss()
        self.value_model_optim = Adam(self.value_model.parameters())
        
        self.reward_model = RewardModel2()
        # self.reward_model = RewardModel(GameRepresentation.PLAY_DESC)
        # self.reward_model_loss = nn.MSELoss()
        # self.reward_model_optim = Adam(self.reward_model.parameters(), lr=1e-1)
        # self.reward_model_sched = ExponentialLR(self.reward_model_optim, gamma=.1)
        
        self.gamma = .1
        
        self.last_state = None
        self.last_action = None
        self.predicted_state = None
        self.objectives = objectives
    
    def _get_dl_from_memory(self, list_states, rewarder, batch_size=9999):
        list_r_disc = np.empty(len(list_states) - 1)
        list_r_real = np.empty(len(list_states) - 1)
        
        with torch.no_grad():
            # on estime le reward restant avec V
            r_disc = torch.zeros(1)
            # r_disc = self.value_model(list_states[-1].get_vector_full().unsqueeze(0)).squeeze(0).detach()
            
            for i in reversed(range(len(list_states) - 1)):
                r_real = rewarder(list_states[i + 1], list_states[i])
                r_disc = r_real + self.gamma * r_disc
                list_r_disc[i] = r_disc
                list_r_real[i] = r_real
            
            states_vectors = [s.get_vector_full().detach() for s in list_states]
            player_vectors = [s.get_player_vector().detach() for s in list_states]
            
            data = zip(states_vectors[:-1],
                       player_vectors[:-1],
                       player_vectors[1:],
                       list_r_real,
                       list_r_disc)
            
            dl = torch.utils.data.DataLoader(list(data), batch_size=batch_size)
        
        # print(np.mean(list_r), np.min(list_r), np.max(list_r))
        return dl
    
    def train_value_model(self, list_states, rewarder):
        dl = self._get_dl_from_memory(list_states, rewarder)
        ll1 = []
        # ll2 = []

        for x, p0, p1, r_real, r_disc in dl:
            self.reward_model.fit((p0, p1), r_real)
            
        for epoch in range(1, 100):
            for x, p0, p1, r_real, r_disc in dl:
                # r_real_pred = self.reward_model(p0, p1)
                # l_real = self.reward_model_loss(r_real_pred, r_real)
                # self.reward_model_optim.zero_grad()
                # l_real.backward()
                # self.reward_model_optim.step()
                
                r_disc_pred = self.value_model(x)
                l_disc = self.value_model_loss(r_disc_pred, r_disc)
                self.value_model_optim.zero_grad()
                l_disc.backward()
                self.value_model_optim.step()
                
                ll1.append(l_disc)
                plt.plot(ll1, label="state value")
                # ll2.append(l_real)
                # plt.plot(np.log10(ll2), label="direct reward")
                plt.legend()
                plt.draw()
                plt.pause(1e-6)
                plt.clf()
        
        # for x, p0, p1, r_real, r_disc in dl:
        #     m = r_real != 0
        #     r_real_pred = self.reward_model(p0, p1)
        #     l_real = self.reward_model_loss(r_real_pred, r_real)
        #     print(l_real.item())
        #     print(p0[m], p1[m], r_real_pred[m], r_real[m])
    
    def _train_world_model(self, action_transitions, test, writer=None):
        ll_map = {action: [] for action in action_transitions}
        ll_play = {action: [] for action in action_transitions}
        ll_test = {action: [] for action in action_transitions}
        
        for action, transitions in action_transitions.items():
            print("learning action", action)
            x_all = torch.stack([s0.get_vector(pos) for s0, pos, _s in transitions])
            x_train, x_test = torch.split(x_all, int(x_all.shape[0] * (1 - test)))
            y_all = torch.stack([s1.get_vector(pos) for _s, pos, s1 in transitions])
            y_train, y_test = torch.split(y_all, int(y_all.shape[0] * (1 - test)))
            
            loss_old = [self.warmup_loss_convergence_eps * i for i in range(5)]
            
            for epoch in range(self.warmup_max_iter):
                if loss_old[-1] < self.model_max_loss:
                    diff = sum(abs(loss_old[i] - loss_old[i + 1]) for i in range(len(loss_old) - 1))
                    if diff < self.warmup_loss_convergence_eps:
                        break
                
                y_pred = self.model.forward_vec(action, v_in=x_train)
                l, (l_map, l_play) = self.model_loss(y_pred, y_train)
                
                self.model_optim.zero_grad()
                l.backward()
                self.model_optim.step()
                self.model_sched.step(epoch // 200)
                
                loss_old.pop(0)
                loss_old.append(l.item())
                
                self.model.eval()
                y_pred_test = self.model.forward_vec(action, v_in=x_test)
                l_test, _details = self.model_loss(y_pred_test, y_test)
                ll_test[action].append(l_test.item())
                self.model.train()
                
                ll_map[action].append(l_map)
                ll_play[action].append(l_play)
                
                if writer is not None:
                    losses = int_loss(x_train, y_pred, y_train, verbose=0)  #trg_idx=a_2_iOut[action]
                    losses["BCE_map"] = l_map.item()
                    losses["MSE_player"] = l_play.item()
                    writer.add_scalars("loss", losses, epoch)
                
                # for a in ll_map:
                # plt.plot(range(len(ll_map[a])), np.log10(ll_map[a]), label=a + "map")
                # plt.plot(range(len(ll_play[a])), np.log10(ll_play[a]), label=a + "play")
                for a in ll_test:
                    plt.plot(range(len(ll_test[a])), np.log10(ll_test[a]), label=a + "t")
                
                plt.legend()
                plt.draw()
                plt.pause(1e-6)
                plt.clf()
            
            else:
                print("warning model did not finish to learn action", action)
            print(epoch, "iteration", "last loss train %.2f %.2f" % (loss_old[-2], loss_old[-1]))
            print(int_loss(x_test, y_pred_test, y_test, verbose=1))
            GameRepresentation.is_correct_vector(y_pred_test, raise_=True)
    
    def learn(self, list_states, list_actions, list_pos, rewarder, test=.3, writer=None):
        # for x in list_states:
        #     x.check()
        
        action_transitions = {}
        for i, action in enumerate(list_actions):
            if action not in action_transitions:
                action_transitions[action] = []
            action_transitions[action].append((list_states[i], list_pos[i], list_states[i + 1]))
        
        plt.ion()
        self._train_world_model(action_transitions, test, writer=writer)
        plt.savefig("world_model_training")
        plt.close()
    
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
        def r(x1, x2):
            return (self.reward_model(x1, x2)).item()
        
        def v(x):
            return self.value_model(x).item()
        
        with torch.no_grad():
            n = bfs.bfs(self.model, self.action_space, self.last_state, r, v, *self.objectives)
        assert n[4] is not None, n[4]
        while n[4][4] is not None:
            # print(n[4])
            n = n[4]
        
        # print(n[4])
        
        assert n[5] is not None
        self.last_action = n[5]
        # print(n)
        self.predicted_state = n[3]
        # exit()
        return self.last_action
    
    def get_result(self, game, reward):
        new_state = GameRepresentation.create_representation_from_game(game)
        
        # self.update(new_state, reward) #todo
        
        if self.predicted_state != new_state:
            y_pred = self.predicted_state.obtained_from_vector
            y_true = new_state.get_vector(self.predicted_state.obtained_from_coo)
            # noinspection PyTypeChecker
            if torch.any((y_pred + .5).int() != y_true) or \
                    np.any(new_state.player_state != self.predicted_state.player_state):
                l, _details = self.model_loss(y_pred.unsqueeze(0), y_true.unsqueeze(0))
                print("learnable prediction error, l=", l.item())
                
                # m_true = new_state.get_vector(self.predicted_state.obtained_from_coo)\
                #     [:-GameRepresentation.PLAY_DESC].reshape((-1, GameRepresentation.TILE_DESC))
                # m_pred = self.predicted_state.get_vector(self.predicted_state.obtained_from_coo)\
                #     [:-GameRepresentation.PLAY_DESC].reshape((-1, GameRepresentation.TILE_DESC))
                # print(tens(m_true-m_pred))
                # print(tens(m_true))
                
                self.model_optim.zero_grad()
                l.backward()
                self.model_optim.step()
        
        self.last_state = new_state
    
    def reset(self, game):
        self.last_state = GameRepresentation.create_representation_from_game(game)
        self.last_action = None
