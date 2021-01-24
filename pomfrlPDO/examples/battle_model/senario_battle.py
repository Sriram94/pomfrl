import random
import math
import numpy as np
from scipy.stats import dirichlet
from scipy.stats import gamma

def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    leftID = random.randint(0, 1)
    rightID = 1 - leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


def play(env, n_round, map_size, max_steps, handles, models, pomfq_position, pomfqv1_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #This function pertains to all algorithms except POMFQ
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    nums_per = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])
        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                temp_list = env.get_neighbors(k,totalagentpositions)
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list

                
                k = k + 1
        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation(handles[i], handles, listofneighbors0, grouplist0, maxelements))
            else:
                state[i] = list(env.get_observation(handles[i], handles, listofneighbors1, grouplist1, maxelements))
        
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids = ids[i])        
        flag = 1
        act_dict = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]
        
        if train:
            models[0].flush_buffer(**buffer)
        
        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob'] = former_act_prob[1]
        
        if train:
            models[1].flush_buffer(**buffer)
        
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                if(len(temp_list) == 0):
                    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                else:
                    temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                new_act_prob[i].append(temp_var[0])

            former_act_prob[i] = np.asarray(new_act_prob[i])
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards




def play2(env, n_round, map_size, max_steps, handles, models, pomfq_position, pomfqv1_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #This functions pertains to POMFQ
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    nums_per = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    lambda_mean = [np.zeros((1, 1)), np.zeros((1, 1))] 
    categories = []
    alphas = {}
    gamma_a = {}
    gamma_b = {}
    for i in range(n_action[0]):
        categories.append(i)
    #simplex = []
    #for i in range(len(categories)):
    #    simplex.append(categories[i]/sum(categories))
    for i in range(sum(nums)):
        tempe_list = [1] * len(categories)
        alphas[i] = tempe_list
        gamma_a[i] = 1
        gamma_b[i] = 1
    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        totalagentpositions = []
        totalagentids = []
        dictofagentposition = {}
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])
                dictofagentposition[identity[i][j]] = positions[i][j]
        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                temp_list = env.get_neighbors(k,totalagentpositions)
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list

                
                k = k + 1
        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation(handles[i], handles, listofneighbors0, grouplist0, maxelements))
            else:
                state[i] = list(env.get_observation(handles[i], handles, listofneighbors1, grouplist1, maxelements))
        
        
        for i in range(n_group):
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
                lambda_mean[i] = np.tile(lambda_mean[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob0=former_act_prob[i], prob1=lambda_mean[i], eps=eps)
        
        flag = 1
        act_dict = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob0'] = former_act_prob[0]
        buffer['prob1'] = lambda_mean[0]
        if train:
            models[0].flush_buffer(**buffer)
        
        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob0'] = former_act_prob[1]
        buffer['prob1'] = lambda_mean[1]
        
        if train:
            models[1].flush_buffer(**buffer)
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])

        
        new_act_prob = [[] for i in range(n_group)]
        new_lambda = [[] for i in range(n_group)]
        
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                length = len(temp_list)
                tempe_list = alphas[ids[i][j]]
                for k in range(length):
                    tempe_list[temp_list[k]] = tempe_list[temp_list[k]] + 1
                tempe_list = np.asarray(tempe_list)
                alphas[ids[i][j]] = tempe_list
                new_samples = dirichlet.rvs(tempe_list, size = 100, random_state = 1)
                new_mean = np.mean(new_samples, axis = 0)
                new_act_prob[i].append(new_mean)
                position_list = []
                if i==0:
                    neigh_list = listofneighbors0[ids[i][j]]
                elif i == 1: 
                    neigh_list = listofneighbors1[ids[i][j]]

                for k in range(len(neigh_list)):
                    position_list.append(dictofagentposition[neigh_list[k]])
                position = dictofagentposition[ids[i][j]]
                a = gamma_a[ids[i][j]]
                b = gamma_b[ids[i][j]]
                scale = 1/b
                r = gamma.rvs(a,scale,100)
                theta = np.mean(r)
                temp_a1, temp_b1, temp_a2, temp_b2 = env.new_gamma(a, b, theta, position, position_list)
                new_a = (temp_a1 + temp_a2)/2
                new_b = (temp_b1 + temp_b2)/2
                gamma_a[ids[i][j]] = new_a 
                gamma_b[ids[i][j]] = new_b
                scale = 1/new_b
                r = gamma.rvs(new_a,scale,100)
                place_holder = []
                place_holder.append(np.mean(r))
                new_lambda[i].append(place_holder)
            
            former_act_prob[i] = np.asarray(new_act_prob[i])
            lambda_mean[i] = np.asarray(new_lambda[i])

        # stat info
        nums = [env.get_num(handle) for handle in handles]
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards



def battle(env, n_round, map_size, max_steps, handles, models, pomfq_position, pomfqv1_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #This function pertains to faceoff battles
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    nums_per = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    former_act_prob2 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
    lambda_mean = [np.zeros((1, 1)), np.zeros((1, 1))]
    categories = []
    alphas = {}
    gamma_a = {}
    gamma_b = {}
    for i in range(n_action[0]):
        categories.append(i)
    for i in range(sum(nums)):
        tempe_list = [1] * len(categories)
        alphas[i] = tempe_list
        gamma_a[i] = 1
        gamma_b[i] = 1

    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        totalagentpositions = []
        totalagentids = []
        dictofagentposition = {}
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])
                dictofagentposition[identity[i][j]] = positions[i][j]
        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                temp_list = env.get_neighbors(k,totalagentpositions)
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list

                
                k = k + 1
        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation(handles[i], handles, listofneighbors0, grouplist0, maxelements))
            else:
                state[i] = list(env.get_observation(handles[i], handles, listofneighbors1, grouplist1, maxelements))
        
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
                former_act_prob2[i] = np.tile(former_act_prob2[i], (len(state[i][0]), 1))
                lambda_mean[i] = np.tile(lambda_mean[i], (len(state[i][0]), 1))
            if i == pomfq_position:
                acts[i] = models[i].act(state=state[i], prob0=former_act_prob2[i], prob1=lambda_mean[i], eps=eps, ids=ids[i])

            elif i == pomfqv1_position: 
                acts[i] = models[i].act(state=state[i], prob=former_act_prob2[i], eps=eps, ids=ids[i])

            else:
                acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids=ids[i])
        
        flag = 1
        act_dict = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]
        
        if train:
            models[0].flush_buffer(**buffer)
        
        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob'] = former_act_prob[1]
        
        if train:
            models[1].flush_buffer(**buffer)
        
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                if(len(temp_list) == 0):
                    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                else:
                    temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                new_act_prob[i].append(temp_var[0])

            former_act_prob[i] = np.asarray(new_act_prob[i])
        
        
        
        new_act_prob = [[] for i in range(n_group)]
        new_lambda = [[] for i in range(n_group)]

        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                length = len(temp_list)
                tempe_list = alphas[ids[i][j]]
                for k in range(length):
                    tempe_list[temp_list[k]] = tempe_list[temp_list[k]] + 1
                tempe_list = np.asarray(tempe_list)
                alphas[ids[i][j]] = tempe_list
                new_samples = dirichlet.rvs(tempe_list, size = 100, random_state = 1)
                new_mean = np.mean(new_samples, axis = 0)
                new_act_prob[i].append(new_mean)
                position_list = []
                if i==0:
                    neigh_list = listofneighbors0[ids[i][j]]
                elif i == 1:
                    neigh_list = listofneighbors1[ids[i][j]]

                for k in range(len(neigh_list)):
                    position_list.append(dictofagentposition[neigh_list[k]])
                position = dictofagentposition[ids[i][j]]
                a = gamma_a[ids[i][j]]
                b = gamma_b[ids[i][j]]
                scale = 1/b
                r = gamma.rvs(a,scale,100)
                theta = np.mean(r)
                temp_a1, temp_b1, temp_a2, temp_b2 = env.new_gamma(a, b, theta, position, position_list)
                new_a = (temp_a1 + temp_a2)/2
                new_b = (temp_b1 + temp_b2)/2
                gamma_a[ids[i][j]] = new_a
                gamma_b[ids[i][j]] = new_b
                scale = 1/new_b
                r = gamma.rvs(new_a,scale,100)
                place_holder = []
                place_holder.append(np.mean(r))
                new_lambda[i].append(place_holder)

            former_act_prob2[i] = np.asarray(new_act_prob[i])
            lambda_mean[i] = np.asarray(new_lambda[i])
        
        
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards

    
    
    
    
    
    
