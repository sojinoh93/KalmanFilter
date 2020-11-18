import numpy as np
import matplotlib.pyplot as plt
import math

from kalmanfilter import KalmanFilter

plt.rcParams["figure.figsize"] = (16,10)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True
plt.legend(fontsize='x-large')
plt.axis("equal")

data = np.genfromtxt('robot-sensor-data1.csv', delimiter='\t', dtype=float, encoding='UTF-8')


uwb = data[1:, 0:2]
gt = data[1:, -2:]

Vx = data[2:, 2] - data[1:-1, 2]
Vy = data[2:, 3] - data[1:-1, 3]
Vw = data[2:, 4] - data[1:-1, 4]
Vg = data[2:, 5] - data[1:-1, 5]
Vg[0] = 0.0

state_odom = uwb[2]
half_pi = math.pi / 2.
diff = uwb[50] - uwb[2]
theta = math.atan2(diff[1], diff[0])

# Kalman filter
F = np.array([
    [1., .0, .0, .0],
    [.0, 1., .0, .0],
    [.0, .0, 1., .0],
    [.0, .0, .0, .0]
])

H = np.array([
    [1., .0, .0, .0],
    [0., 1., .0, .0],
    [1., .0, .0, .0],
    [0., 1., .0, .0]
])

R = np.diag([1., 1., np.deg2rad(0.1), 1.])**2  # State covariance
Q = np.diag([1.5, 1.5, 1.5, 1.5])**2  # Observation covariance

x_est = np.array([
    [uwb[1][0]],
    [uwb[1][1]],
    [theta],
    [.0]
])  # [x, y, theta, v]

P_est = np.eye(4)

v = 1.0  # m/s
w = 1.0  # rad/s
u = np.array([[v, w]]).T
kf = KalmanFilter(F, H, Q, R)

# History
h_odom = state_odom
h_u = uwb[2]
h_gt = gt[2]
h_x_est = x_est.T

max_err = 0.
min_err = 10000.

for i in range(len(Vx)):
    # Update Wx, Vx, Vy
    theta = theta + np.radians(Vg[i])
    state_odom = state_odom[0] + Vx[i] * math.cos(theta), state_odom[1] + Vx[i] * math.sin(theta)
    state_odom = state_odom[0] + Vy[i] * math.cos(theta-half_pi), state_odom[1] + Vy[i] * math.sin(theta-half_pi)

    z = np.array([
        [uwb[i][0]],
        [uwb[i][1]],
        [state_odom[0]],
        [state_odom[1]]
    ])

    x_est, P_est = kf.fit2(x_est, P_est, z)

    dist = x_est[0:2].T[0] - gt[i]
    err = np.sqrt(dist[0] ** 2 + dist[1] ** 2)
    err_kf = np.sqrt(dist[0] ** 2 + dist[1] ** 2)

    dist_uwb = uwb[i] - gt[i]
    err_uwb = np.sqrt(dist_uwb[0] ** 2 + dist_uwb[1] ** 2)

    if (err > max_err):
        max_err = err
    if (err < min_err):
        min_err = err

    # print("Max error: {}\t Min error: {}".format(max_err, min_err))
    print("KF error: {} \t UWB error: {}".format(err_kf, err_uwb))

    # History
    h_odom = np.vstack((h_odom, state_odom))
    h_u = np.vstack((h_u, uwb[i]))
    h_gt = np.vstack((h_gt, gt[i]))
    h_x_est = np.vstack((h_x_est, x_est.T))

    plt.cla()
    plt.plot(h_odom[:, 0], h_odom[:, 1], '.k', label='odometry')
    plt.plot(h_u[:, 0], h_u[:, 1], '.r', label='uwb')
    plt.plot(h_gt[:, 0], h_gt[:, 1], '.b', label='point cloud')
    plt.plot(h_x_est[:, 0], h_x_est[:, 1], '-g', label='kalman filter')
    plt.pause(0.001)
