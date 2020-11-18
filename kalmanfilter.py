import numpy as np

class KalmanFilter():
    def __init__(self, F, H, Q, R):
        # State model matrice.
        self.F = F
        self.H = H

        # Convariance matrice. Control Params
        self.R = R
        self.Q = Q

    def fit(self, x_est, P_est, B, u, z):
        # 1. Predict
        x_pred = self.F.dot(x_est) + B.dot(u)                       # x' = Fx + Bu
        P_pred = self.F.dot(P_est).dot(self.F.T) + self.R           # P' = FPtFt + R

        # 2. Update
        z_pred = self.H.dot(x_pred)                                 # z' = Hx
        y = z - z_pred                                              # y = z - z'
        S = self.H.dot(P_pred).dot(self.H.T) + self.Q               # S = HP'Ht + Q
        K = P_pred.dot(self.H.T).dot(np.linalg.inv(S))              # K = P'HtInv(S)
        x_est = x_pred + K.dot(y)                                   # x = x' + Ky
        P_est = (np.eye(len(x_est)) - K.dot(self.H)).dot(P_pred)    # P = (I-KH)P'
        
        return x_est, P_est

    def fit2(self, x_est, P_est, z):
        # 1. Predict
        x_pred = self.F.dot(x_est)  # + B.dot(u)                       # x' = Fx + Bu
        P_pred = self.F.dot(P_est).dot(self.F.T) + self.R  # P' = FPtFt + R

        # 2. Update
        z_pred = self.H.dot(x_pred)  # z' = Hx
        y = z - z_pred  # y = z - z'
        S = self.H.dot(P_pred).dot(self.H.T) + self.Q  # S = HP'Ht + Q
        K = P_pred.dot(self.H.T).dot(np.linalg.inv(S))  # K = P'HtInv(S)
        x_est = x_pred + K.dot(y)  # x = x' + Ky
        P_est = (np.eye(len(x_est)) - K.dot(self.H)).dot(P_pred)  # P = (I-KH)P'

        return x_est, P_est