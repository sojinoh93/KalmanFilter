# The code is based on PythonRobotics (https://github.com/AtsushiSakai/PythonRobotics#extended-kalman-filter-localization)
import numpy as np
import math
import matplotlib.pyplot as plt

from kalmanfilter import KalmanFilter
from plot import plot_covariance_ellipse

dt = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

F = np.array([
    [1., .0, .0, .0],
    [.0, 1., .0, .0],
    [.0, .0, 1., .0],
    [.0, .0, .0, .0]
])

H = np.array([
    [1., .0, .0, .0],
    [0., 1., .0, .0]
])

R = np.diag([0.1, 0.1, np.deg2rad(1.0), 0.1])**2  # predict state covariance
Q = np.diag([1.5, 1.5])**2  # Observation x,y position covariance

def simulate(x_gt, F, B, u):
    x_gt = F.dot(x_gt) + B.dot(u)

    Qsim = np.diag([0.5, 0.5]) ** 2
    zx = x_gt[0, 0] + np.random.randn() * Qsim[0, 0]
    zy = x_gt[1, 0] + np.random.randn() * Qsim[1, 1]

    z = np.array([
        [zx],
        [zy]
    ])
    return x_gt, z

def main():
    time = 0.0
    x_est = np.zeros((4, 1))
    x_gt = np.zeros((4, 1))
    P_est = np.eye(4)

    v = 1.0  # m/s
    w = 0.1  # rad/s
    u = np.array([[v, w]]).T

    kf = KalmanFilter(F, H, Q, R)

    # history
    h_x_est = x_est
    h_x_gt = x_gt
    h_z = np.zeros((1, 2))

    while SIM_TIME >= time:
        time += dt
        B = np.array([
            [dt * math.cos(x_est[2, 0]), .0],
            [dt * math.sin(x_est[2, 0]), .0],
            [.0, dt],
            [1., .0]
        ])
        x_gt, z = simulate(x_gt, F, B, u)
        x_est, P_est =kf.fit(x_est,P_est, B, u, z)

        # store data history
        h_x_est = np.hstack((h_x_est, x_est))
        h_x_gt = np.hstack((h_x_gt, x_gt))
        h_z = np.vstack((h_z, z.T))

        plt.cla()
        plt.plot(h_z[:, 0], h_z[:, 1], ".g")
        plt.plot(h_x_gt[0, :].flatten(),
                 h_x_gt[1, :].flatten(), "-b")
        plt.plot(h_x_est[0, :].flatten(),
                 h_x_est[1, :].flatten(), "-r")
        plot_covariance_ellipse(x_est, P_est)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

if __name__ == '__main__':
    main()