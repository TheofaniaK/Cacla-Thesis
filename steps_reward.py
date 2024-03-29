import numpy as np
from force_feedback import force

checkpoints = [(6/20)*(i+1) for i in range(20)]

def init():
    global checkpoints
    checkpoints = [(6 / 20) * (i + 1) for i in range(20)]

def get_reward(x, pen_new, pen_previous, motion_dir, di, c):
    global checkpoints
    time = -1
    go_right = -0.5
    if motion_dir > 0:
        go_right = 0.5
    diff = pen_previous - pen_new
    if diff >= 1:
        c += 1
    di.append(diff)
    # print('Previous Penetration =', pen_previous)
    # print('New Penetration =', pen_new)
    # print('Movement Direction =', motion_dir)
    # print('Differential Position =', diff)

    go_right_weight = 2
    diff_weight = 200
    pen_weight = 10
    time_weight = 1

    #+ go_right_weight * go_right
    #+ time_weight * time

    before = len(checkpoints)
    checkpoints = [i for i in checkpoints if i > x]
    after = len(checkpoints)

    if diff > 0:  # out of the corridor but directed towards it
        #print('IF #1')
        if pen_new == 0:
            if go_right > 0:
                reward = 4 * diff_weight * diff + go_right_weight * go_right
            else:
                reward = -2
        #elif pen_new <= 1:
        else:
            if go_right > 0:
                reward = 4 * pen_weight * pen_new + go_right_weight * go_right
            else:
                reward = -2
    elif diff < 0:  # out of the corridor and moving away from it
        #print('IF #2')
        if pen_previous == 0:
            reward = diff_weight * diff #+ go_right_weight * go_right
        #elif pen_new <= 1:
        else:
            reward = -20 * pen_weight * pen_new #+ go_right_weight * go_right
    else:  # diff == 0
        if go_right < 0:
            reward = -2  #5 + go_right_weight * go_right
        else:
            reward = 20  #+ go_right_weight * go_right
        #print('IF #3')

    if reward > 25:
        reward = 25
    elif reward < -25:
        reward = -25

    # if 0 < reward < 10:
    #     reward = 10
    # elif 10 <= reward < 20:
    #     reward = 15
    # elif reward >= 20:
    #     reward = 20
    # elif reward <= -20:
    #     reward = -20
    # elif -20 < reward <= -10:
    #     reward = -15
    # else:
    #     reward = -10

    # reward += 10 * (before - after)

    #print('Reward is:', reward, np.shape(reward))
    #print('<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>')

    return reward, di, c


def take_step(action, x, y, radi, coord1, coord2, penetration_prev, pen_init, motion_dir, init_pos, dif, co, problem):
    """
    makes a step.
    returns new observation, reward, whether episode is done or not and (optional) additional info.
    for more information see OpenAI Gym documentation.
    """

    done = False
    # limit the joint actions to interval [-1, 1] ...
    action = np.array(action)
    # action[action > 1.0] = 1.0
    # action[action < -1.0] = -1.0

    #print('<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>')
    # when using relative actions, joint_positions1 = joint_positions0 + (action * action_multiplier)
    position0 = [[x[0], y[0]]]
    # position1 = position0 + action
    # position1[position1 > 1.0] = 1.0
    # position1[position1 < 0] = 0

    # if np.sqrt(np.sum(action ** 2)) == 0.:
    #     normalized_action = action
    # elif np.sqrt(np.sum(action ** 2)) > 0.1:
    #     normalized_action = action / 10 / np.sqrt(np.sum(action ** 2))
    # else:
    #     normalized_action = action

    if np.sqrt(np.sum(action ** 2)) == 0.:
        normalized_action = action
    else:
        normalized_action = action / 10 / np.sqrt(np.sum(action ** 2))

    position1 = position0 + normalized_action

    # print('position0 is:', position0, np.shape(position0))
    # print('action is:', action, np.shape(action))
    print('norm_action is:', normalized_action, np.shape(normalized_action))
    #norm_pos = position1 / np.sqrt(np.sum(position1 ** 2))

    #print('position1 is:', position1, np.shape(position1))

    ################# ??????????????????? ###################
    # self.iteration_n += 1

    # calculate return values

    print('TAKE STEP X AND Y:', position1[0][0], np.shape(position1[0][0]), position1[0][1], np.shape(position1[0][1]))

    f2 = force(position1[0][0], position1[0][1], float(radi), coord1, coord2)
    print('values of f in take step are:', f2[0], np.shape(f2[0]), f2[1], np.shape(f2[1]), f2[2], np.shape(f2[2]))
    penetration_new = f2[0]
    observation = f2[2]
    #print('Observation in take step is:', observation, np.shape(observation))

    if penetration_new > 0.15:
        done = True
        problem += 1
        # norm_pos = init_pos
        # print('####### PROBLEM #######')
        # #print('position1 =', position1, np.shape(position1))
        # f3 = force(norm_pos[0][0], norm_pos[0][1], float(radi), coord1, coord2)
        # penetration_new = f3[0]
        # observation = f3[1]
        # penetration_prev = pen_init
        # motion_dir = -1
        # #print('Observation in take step  problem is:', observation, np.shape(observation))

    #print('<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>')

    reward, dif, co = get_reward(x[0], penetration_new, penetration_prev, motion_dir, dif, co)

    #print('observation in take step is:', observation)
    # distance = self.get_distance()
    # rd = 1 - 2 * (distance / self.max_distance)

    x = position1[0][0]
    y = position1[0][1]

    if (x - float(radi) > coord1[-1, 0] and y - float(radi) > coord1[-1, 1] and y + float(radi) < coord2[-1, 1]) or \
            (x + float(radi) < coord1[0, 0] and y - float(radi) > coord1[0, 1] and y + float(radi) < coord2[0, 1]):
        # print('cx - rad = ', x - float(rad), ', coords1[-1,0] = ', coords1[-1, 0])
        # print('coords2[-1,1] = ', coords2[-1, 1], ', cy - rad = ', y - float(rad), ', coords1[-1,1] = ', coords1[-1, 1])
        done = True
    # done = True if distance < 0.01 or self.iteration_n > 100 else False

    return observation, reward, done, [x], [y], penetration_new, dif, co, problem, action

def take_step_test(action, x, y, r, coord1, coord2, penetration_prev, motion_dir):

    done = False
    action = np.array(action)

    position0 = [[x[0], y[0]]]

    # if np.sqrt(np.sum(action ** 2)) == 0.:
    #     normalized_action = action
    # elif np.sqrt(np.sum(action ** 2)) > 0.1:
    #     normalized_action = action / 10 / np.sqrt(np.sum(action ** 2))
    # else:
    #     normalized_action = action

    if np.sqrt(np.sum(action ** 2)) == 0.:
        normalized_action = action
    else:
        normalized_action = action / 10 / np.sqrt(np.sum(action ** 2))

    #normalized_action = action / 10 / np.sqrt(np.sum(action ** 2))
    position1 = position0 + normalized_action

    # action[action > 1.0] = 1.0
    # action[action < -1.0] = -1.0

    # print('<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><>')
    # when using relative actions, joint_positions1 = joint_positions0 + (action * action_multiplier)
    # position1 = position0 + action
    # position1[position1 > 1.0] = 1.0
    # position1[position1 < 0] = 0

    f5 = force(position1[0][0][0], position1[0][0][1], float(r), coord1, coord2)
    penetration_new = f5[0]
    observation = f5[2]

    reward, dif, co = get_reward(x[0], penetration_new, penetration_prev, motion_dir, [], 0)

    x = position1[0][0][0]
    y = position1[0][0][1]

    # posx = coord1[-1, 0]
    # posy = (coord1[-1, 1] + coord2[-1, 1]) / 2

    if x - float(r) > coord1[-1, 0] and y - float(r) > coord1[-1, 1] and y + float(r) < coord2[-1, 1]:
        done = True

    dist = np.zeros(3)

    dist[0] = x - float(r) - coord1[-1, 0]
    dist[1] = y - float(r) - coord1[-1, 1]
    dist[2] = y + float(r) - coord2[-1, 1]

    return observation, reward, done, [x], [y], penetration_new, normalized_action, dist