from builtins import AssertionError
import heapq
import numpy as np


def bfs(model, action_space, s0, R, V, t, is_goal, max_iter=150):
    # cum_r + expected_value, cum_r, depth, state, noeud précédent, action_précédente
    opened_states = [(np.inf, 0, 0, s0, None, None, 0)]
    visited_states = {opened_states[0]: np.inf}
    
    best_node = opened_states[0]
    
    dbg_tree = {}
    dbg_node_list = [opened_states[0]]
    
    found = False
    for i in range(max_iter + 1):
        if not len(opened_states):
            break
        node = heapq.heappop(opened_states)
        
        vr, r, d, s, prev_node, prev_a, _id = node
        dbg_tree[_id] = []
        # print(len(visited_states), len(opened_states), -best_node[0], d, prev_a)
        
        if is_goal(s, t):
            print("goal found", i, vr, s.player_state)
            return node
        
        if vr < best_node[0] or best_node[3] is None:
            best_node = node
        
        if i == max_iter:
            break
        
        d2 = d + 1
        # print()
        for xi in range(s.map_state.shape[0]):
            for yi in range(s.map_state.shape[1]):
                if s.map_state[xi][yi][5] == 1 and s.map_state[xi][yi][7] == 1:
                    for a in action_space:
                        try:
                            s2 = model.forward_full(s, (xi, yi), a)
                        except AssertionError:
                            print("astar error")
                            print(s.map_state.shape)
                            for x in s.map_state:
                                print(x)
                            print(s.player_state)
                            model.forward_full(s, (xi, yi), a)
                            assert 0
                        
                        
                        r2 = r + R(s.get_player_vector(), s2.get_player_vector())
                        v2 = V(s2.get_vector_full())
                        vr2 = -v2 - r2
                        
                        # état déjà vu, et on ne fait pas mieux
                        if s2 in visited_states and vr2 >= visited_states[s2]: continue
                        
                        # print("\t", a, v2)
                        
                        found = max(found, r2)
                        
                        nnode = (vr2, r2, d2, s2, node, ((xi, yi), a), len(dbg_node_list))
                        
                        dbg_tree[_id].append(len(dbg_node_list))
                        dbg_node_list.append(nnode)
                        
                        heapq.heappush(opened_states, nnode)
                        visited_states[s2] = vr2
    
    def f(id_, cpt=-1):
        if cpt != -1:
            x = dbg_node_list[id_]
            print("\t" * cpt, "%.2f" % -x[0], "%.2f" % x[1])
        if id_ not in dbg_tree:
            return
        for id2 in dbg_tree[id_]:
            f(id2, cpt + 1)
    
    # f(0)
    print('found', found)
    # print()
    
    
    assert best_node is not None
    return best_node
