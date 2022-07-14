import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import sys
from cacla import Cacla
from force_feedback import force
"""tf.test.is_gpu_available()"""
from sim_file import simulation
from steps_reward import take_step, take_step_test

def run_episode(model, x, y, radi, coord1, coord2):
    """
    The core of training.
    For each movement (until the variable done != True) calculates value function at time T0 and T1,
    based on explored action a0 ~ A0 + exploration.
    Fits critic and actor according to learning rule.
    Saves each step into variable trajectory, and at the end returns full list of steps.
    """

    # initialize variables and reset environment
    trajectory = []
    V = []
    Vf = []
    rew = []
    d = []
    delt = []
    initial = [[x[0], y[0]]]
    model.critic.save_weights('model.h5')
    model.actor.save_weights('model.h6')
    #print('initial is:', initial, np.shape(initial))
    f1 = force(x, y, float(radi), coord1, coord2)
    pen_prev = f1[0]
    pen_init = pen_prev
    observation0 = f1[1]
    done = False
    problem = 0
    i = 0
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(1, 1, 1)
    # ax1.axis('equal')
    # ax1.plot(coord1[:, 0], coord1[:, 1])
    # ax1.plot(coord2[:, 0], coord2[:, 1])
    counter = 0
    update = 0
    cnt = 0
    fores = []
    #diaf = []
    while not done:
        i += 1
        # if i % 50 == 0:
        #     plt.cla()
        #     ax1.axis('equal')
        #     ax1.plot(coords1[:, 0], coords1[:, 1])
        #     ax1.plot(coords2[:, 0], coords2[:, 1])
        if i == 200:
            #plt.close(fig1)
            print('       V0,                    V1,                    reward,                  diff,                  delta are:')
            k = 1
            for (j, z, q, x, u) in zip(V, Vf, rew, d, delt):
                print(k,
                      '   ', np.around(j, 4),
                      '              ', np.around(z, 4),
                      '              ', np.around(q, 4),
                      '              ', np.around(x, 4),
                      '              ', np.around(u, 4), end='\n')
                k += 1
            print('Counter is:', counter)
            print('Diff is >= 1:', cnt, 'times')
            print(fores)
            plt.plot(rew)
            plt.pause(0.5)
            print('Total updates are:', update)
            print('Total problems are:', problem)
            # for t in diaf:
            #     print(t, end='\n')
            """j = 0
            V = []
            A = []
            a = []
            while j < 12:
                print('@@@@@@@@@@@@@@', j, ' @@@@@@@@@@@@@@@@')
                # get current value of value function for observation0
                V0 = model.critic.predict(np.array([observation0]))
                V.append(V0)
                # predict default action
                A0 = model.actor.predict(np.array([observation0]))
                A.append(A0)
                # sample new explored action
                a0 = model.sample(A0[0], model.exploration_factor)
                a.append(a0)
                j += 1

            print('V is:')
            for j in V:
                print(j)
            print('------------------------------------')
            print('A is:')
            for j in A:
                print(j)"""
            break
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('%%%%%%%%%%%%%   I =', i, '  %%%%%%%%%%%%%')
        # get current value of value function for observation0
        print('np.array([observation0]) is:', np.array([observation0]), np.shape(np.array([observation0])))
        V0 = model.critic.predict(np.array([observation0]))
        V.append(V0[0])
        print('V0 is:', V0)
        # predict default action
        A0 = model.actor.predict(np.array([observation0]))
        print('A0 is:', A0)
        # sample new explored action
        a0 = model.sample(A0[0], model.exploration_factor)
        print('a0 is:', a0)
        a0 = [a0]
        motion_direction = a0[0][0]
        observation1, reward, done, x, y, penetration, d, cnt_new, action_new, prob = take_step(a0, x, y, float(radi), coord1, coord2, pen_prev, pen_init, motion_direction, initial, d, cnt, problem)
        pen_prev = penetration
        problem = prob
        if cnt_new == cnt + 1:
            fores.append(i)
        cnt = cnt_new
        rew.append(reward)
        # get current value of value function for observation1 and compute delta.
        print('np.array([observation1]) is:', np.array([observation1]), np.shape(np.array([observation1])))
        V1 = model.critic.predict(np.array([observation1]))
        Vf.append(V1[0])
        print('V1 is:', V1)
        delta = reward + model.gamma * V1 - V0
        print('delta is:', delta, np.shape(delta))
        delt.append(delta[0])

       # print('[reward + model.gamma * V1] is:', [reward + model.gamma * V1], np.shape([reward + model.gamma * V1]))
        # fit critic
        model.critic.fit(np.array([observation0]), [reward + model.gamma * V1], batch_size=1, verbose=0)
        #print('DELTA IS: ', delta)

        if reward < 0 and delta > 0:
            counter += 1

        if delta > 0:
            model.actor.fit(np.array([observation0]), np.array(action_new), batch_size=1, verbose=0)
            observation0 = observation1
            update += 1
            print('##### ACTOR UPDATED #####')
        else:
            print('##### ACTOR NOT UPDATED #####')

        if done:
            plt.plot(rew)
            plt.pause(0.5)
            print('Total updates are:', update)
            print('Total problems are:', problem)

        # save and append trajectory.
        # step = np.zeros(2)
        # step[0] = x[0]
        # step[1] = y[0]
        # circle = [plt.Circle((x[0], y[0]), float(radi), color='k', fill=False)]
        # ax1.add_patch(circle[0])
        # ax1.axis('equal')
        # ax1.plot(x[0], y[0], 'o', color='r')
        # plt.pause(0.02)
        # input('Continue?')

        # save and append trajectory.
        step = {"observation0": observation0, "observation1": observation1,
                "V0": V0[0], "V1": V1[0], "A0": A0[0][:], "a0": a0[0][:], "normalized action": action_new[0][:],
                "reward": reward, "delta": delta[0][0]}
        trajectory.append(step)

    return trajectory


def run_batch(model, pt_x, pt_y, ra, coor1, coor2, batch_sz):
    """
    Accepts CACLA model and 'batch size'. Runs number of episodes equal to batch_size.
    Logs the rewards and at the end returns all traversed trajectories.
    """
    trajectories = []
    total_steps = 0

    # run n=batch_size episodes. save trajectories on the way.
    for _ in range(batch_sz):
        trajectory = run_episode(model, pt_x, pt_y, float(ra), coor1, coor2)
        total_steps += len(trajectory)

        trajectories.append(trajectory)

    return trajectories

def train(model, x_dim, y_dim, ra, coord1, coord2, n_episod, batch_sz):
    """
    Accepts model (CACLA in our case), number of all episodes, batch size and optional argument to run GUI.
    Trains the actor and critic for number of episodes.
    """
    episode = 0
    while episode < n_episod:
        run_batch(model, x_dim, y_dim, float(ra), coord1, coord2, batch_sz)
        episode += batch_sz

        # update learning and exploration rates for the algorithm.
        model.update_lr(model.lr_decay)
        model.update_exploration()

def test(model, n, pt_x, pt_y, coor1, coor2, ra):
    """
    tests the model.
    n is the number of locations to test.
    """
    success = 0
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(1, 1, 1)
    ax5.axis('equal')
    ax5.plot(coor1[:, 0], coor1[:, 1])
    ax5.plot(coor2[:, 0], coor2[:, 1])
    for i in range(n):
        print('I is:', i)
        # reset the environment.
        #observation = model.env.reset()
        ft = force(pt_x, pt_y, float(ra), coor1, coor2)
        penetration = ft[0]
        observation = ft[1]
        done = False
        # repeat until done (arm reaches the target / 100 steps).
        while not done:
            # if model.env.simulation:
            #     model.env.render()
            # use actor to predict next action.
            action = model.actor.predict(np.array([observation]))
            motion_dir = action[0][0]

            # make a step.
            observation, reward, done, pt_x, pt_y, pen_new, action_new, distance = take_step_test([action], pt_x, pt_y, float(ra), coor1, coor2, penetration, motion_dir)
            penetration = pen_new
            step = np.zeros(2)
            step[0] = pt_x[0]
            step[1] = pt_y[0]
            circle = [plt.Circle((pt_x[0], pt_y[0]), float(ra), color='k', fill=False)]
            ax5.add_patch(circle[0])
            ax5.axis('equal')
            ax5.plot(pt_x[0], pt_y[0], 'o', color='r')
            plt.pause(0.02)
            print("iteration:", i, "reward:", reward, "distance:", distance, "done:", done)
            # if distance < 0.01:
            if done:
                success += 1
    plt.savefig("test.png")
        # model.env.render()
    print("success rate:", success / n)

######################################################################################################################################################


if __name__ == "__main__":
    point_x, point_y, coordinates1, coordinates2, radius, offset = simulation()
    f = force(point_x, point_y, float(radius), coordinates1, coordinates2)

    # initialize parameters
    input_dim = 2  # env.observation_space.shape[0]
    output_dim = 2  # env.action_space.shape[0]
    alpha = 0.01  # learning rate for actor
    beta = 0.01  # learning rate for critic
    lr_decay = 0.997  # lr decay
    exploration_decay = 0.997  # exploration decay
    gamma = 0.01  # discount factor
    exploration_factor = 0.35

    n_episodes = 1000
    batch_size = 10

    algorithm = Cacla(input_dim, output_dim, alpha, beta, gamma, lr_decay, exploration_decay, exploration_factor)
    #traj = run_episode(algorithm, point_x, point_y, float(radius), coordinates1, coordinates2)
    train(algorithm, point_x, point_y, float(radius), coordinates1, coordinates2, n_episodes, batch_size)
    #input('Continue?')
    algorithm.critic.save_weights('model.h5')
    algorithm.actor.save_weights('model.h6')
    pnt_x, pnt_y, coords1, coords2, rad, offs = simulation()
    test(algorithm, 20, pnt_x, pnt_y, coords1, coords2, rad)
    print('############  REACHED THE END  ############')

    sys.exit()