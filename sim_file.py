import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from scipy.interpolate import interp1d
from scipy import interpolate

def simulation():
    # offset_sim = input("Choose the offset of the corridor: ")
    # rad = input("Choose the radius of the cyclic object: ")
    # step = input("Choose the step of the cuclic object: ")
    mode = input("Choose 0 for creating trajectory, or 1 for existing: ")
    offset_sim = 1
    r_sim = 0.2
    coords1_sim = np.array([])
    coords2_sim = np.array([])

    while (mode != '0') and (mode != '1'):
        mode = input("Wrong! Choose 0 for creating trajectory, or 1 for existing: ")

    if mode == '0':
        while True:
            plt.clf()
            plt.setp(plt.gca(), autoscale_on=False)
            print('You will generate a path in 2d-space, click to begin')
            plt.waitforbuttonpress()

            #track = []
            print('Select some points with the mouse and click "Enter" to finish')
            track = np.asarray(plt.ginput(-1, timeout=-1))
            print(track)

            resolution = 10000
            dof_1_orig = track[:, 0]
            dof_2_orig = track[:, 1]
            tck, u = interpolate.splprep([dof_1_orig, dof_2_orig], s=0)
            u_new = np.arange(0, 1.0 + (1 / resolution), 1 / resolution)
            dof_1, dof_2 = interpolate.splev(u_new, tck)
            dof = np.transpose(np.vstack((dof_1, dof_2)))
            plt.plot(dof[:, 0], dof[:, 1], 'k')
            # plt.plot(dof[:,0], dof[:,1] + float(offset), 'k')
            print('Satisfied? Key click for yes, mouse click for no')
            coords1_sim = np.zeros((len(dof[:, 0]), 2))
            coords1_sim[:, 0] = dof[:, 0]
            coords1_sim[:, 1] = dof[:, 1]

            if plt.waitforbuttonpress():
                """mode_seg = 0
                np.savetxt('datapoints.txt', dof, delimiter=' ', fmt='%1.4f')"""
                break

        while True:
            plt.clf()
            plt.setp(plt.gca(), autoscale_on=False)
            print('You will generate a path in 2d-space, click to begin')
            plt.plot(coords1_sim[:, 0], coords1_sim[:, 1], 'k')
            plt.waitforbuttonpress()

            #track = []
            print('Select some points with the mouse and click "Enter" to finish')
            track = np.asarray(plt.ginput(-1, timeout=-1))
            print(track)

            resolution = 10000
            dof_1_orig1 = track[:, 0]
            dof_2_orig1 = track[:, 1]
            tck, u = interpolate.splprep([dof_1_orig1, dof_2_orig1], s=0)
            u_new = np.arange(0, 1.0 + (1 / resolution), 1 / resolution)
            dof_11, dof_21 = interpolate.splev(u_new, tck)
            dof1 = np.transpose(np.vstack((dof_11, dof_21)))
            plt.plot(dof1[:, 0], dof1[:, 1], 'k')
            print('Satisfied? Key click for yes, mouse click for no')
            coords2_sim = np.zeros((len(dof1[:, 0]), 2))
            coords2_sim[:, 0] = dof1[:, 0]
            coords2_sim[:, 1] = dof1[:, 1]

            if plt.waitforbuttonpress():
                """mode_seg = 0
                np.savetxt('datapoints.txt', dof1, delimiter=' ', fmt='%1.4f')"""
                break
    elif mode == '1':
        b = int(2 * np.pi // 0.5)
        points = np.zeros((b + 1, 3))
        i = 0
        j = 0
        while i < 2 * np.pi:
            points[j, 0] = i
            points[j, 1] = np.sin(i)
            points[j, 2] = np.sin(i) + float(offset_sim)
            i += 0.5
            j += 1

        len_x = len(points[:, 0])

        f1 = interp1d(points[:, 0], points[:, 1], kind='cubic')
        f2 = interp1d(points[:, 0], points[:, 2], kind='cubic')

        x_new = np.linspace(0, b // 2, num=5 * len_x, endpoint=True)
        y_new = f1(x_new)
        y_new2 = f2(x_new)

        coords1_sim = np.zeros((len(x_new), 2))
        coords1_sim[:, 0] = x_new
        coords1_sim[:, 1] = y_new

        coords2_sim = np.zeros((len(x_new), 2))
        coords2_sim[:, 0] = x_new
        coords2_sim[:, 1] = y_new2

    ######################################################################################################################################################
    plt.clf()
    plt.setp(plt.gca(), autoscale_on=False)

    # Create figure and subplot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.plot(coords1_sim[:, 0], coords1_sim[:, 1])
    ax.plot(coords2_sim[:, 0], coords2_sim[:, 1])

    print('You will generate a starting point for the cyclic object, click to begin')
    plt.waitforbuttonpress()

    #track = []
    print('Select one point with the mouse and click "Enter" to finish')
    track = np.asarray(plt.ginput(1, timeout=-1))
    print(track)

    #resolution = 10000
    cx = track[:, 0]
    cy = track[:, 1]

    coord_x = []
    coord_y = []
    circle = []

    coord_x.append(cx)
    coord_y.append(cy)
    #i = 0

    circle.append(plt.Circle((cx, cy), float(r_sim), color='k', fill=False))
    ax.add_patch(circle[0])
    ax.axis('equal')
    ax.plot(cx, cy, 'o', color='k')

    plt.pause(0.5)
    plt.close(fig)

    return cx, cy, coords1_sim, coords2_sim, r_sim, offset_sim