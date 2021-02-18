import numpy as np
import matplotlib.pyplot as plt


def compute_loss(jitter, correction):

    # X translation loss
    jitter_X = np.diff(jitter['X'])[1:]
    correction_X = np.diff(correction['X'])
    loss_X = np.sum(np.abs(jitter_X + correction_X))

    # Y translation loss
    jitter_Y = np.diff(jitter['Y'])[1:]
    correction_Y = np.diff(correction['Y'])
    loss_Y = np.sum(np.abs(jitter_Y + correction_Y))

    # rotation loss
    jitter_R = np.diff(jitter['R'])[1:]
    correction_R = np.diff(correction['R'])
    loss_R = np.sum(np.abs(jitter_R + correction_R))

    return (1/3)*(loss_X+loss_Y+loss_R)


def plot_traj(dict, correction=False):

    if correction:
        delta = 1
    else:
        delta = -1

    plt.figure(figsize=(5, 5))
    X_traj = delta*dict['X']
    Y_traj = delta*dict['Y']
    plt.plot(X_traj, Y_traj, '+-')
    plt.plot([0], [0], 'ro')
    plt.xlabel('translation X')
    plt.ylabel('translation Y')
    plt.show()
