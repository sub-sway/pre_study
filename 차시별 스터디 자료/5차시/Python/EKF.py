import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 설정
dt = 0.1
Q = np.diag([0.01, 0.01, np.deg2rad(1.0)])**2  # 프로세스 잡음
R = np.diag([0.5])**2                         # 측정 잡음
landmark = np.array([5.0, 5.0])               # 고정 랜드마크

# 초기 상태
x_true = np.array([0.0, 0.0, 0.0])            # 실제 위치 [x, y, theta]
x_est = np.array([0.0, 0.0, 0.0])             # 추정 위치
P = np.eye(3)                                 # 추정 오차 공분산

x_log, y_log, est_log = [], [], []

def motion_model(x, u):
    x_new = np.zeros_like(x)
    x_new[0] = x[0] + u[0] * np.cos(x[2]) * dt
    x_new[1] = x[1] + u[0] * np.sin(x[2]) * dt
    x_new[2] = x[2] + u[1] * dt
    return x_new

def observation_model(x, landmark):
    dx = landmark[0] - x[0]
    dy = landmark[1] - x[1]
    return np.sqrt(dx**2 + dy**2)  # 거리만 측정

def jacob_F(x, u):
    theta = x[2]
    jF = np.array([
        [1, 0, -u[0]*np.sin(theta)*dt],
        [0, 1,  u[0]*np.cos(theta)*dt],
        [0, 0, 1]
    ])
    return jF

def jacob_H(x, landmark):
    dx = landmark[0] - x[0]
    dy = landmark[1] - x[1]
    dist = np.sqrt(dx**2 + dy**2)
    return np.array([[-dx/dist, -dy/dist, 0]])

def ekf_estimation(x_est, P, z, u):
    # 예측
    x_pred = motion_model(x_est, u)
    F = jacob_F(x_est, u)
    P_pred = F @ P @ F.T + Q

    # 측정 예측
    z_pred = observation_model(x_pred, landmark)
    H = jacob_H(x_pred, landmark)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # 업데이트
    y = z - z_pred
    x_upd = x_pred + K.flatten() * y
    P_upd = (np.eye(len(x_est)) - K @ H) @ P_pred

    return x_upd, P_upd

# 애니메이션 업데이트 함수
def update(frame):
    global x_true, x_est, P

    u = np.array([1.0, np.deg2rad(10)])  # 속도 1.0, 회전 10도/s
    x_true = motion_model(x_true, u)
    z = observation_model(x_true, landmark) + np.random.randn() * np.sqrt(R[0][0])

    x_est, P = ekf_estimation(x_est, P, z, u)

    x_log.append(x_true[0])
    y_log.append(x_true[1])
    est_log.append(x_est[:2])

    ax.clear()
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.plot(x_log, y_log, 'b-', label='True Path')
    ax.plot(*zip(*est_log), 'r--', label='EKF Estimate')
    ax.scatter(*landmark, c='g', label='Landmark')
    ax.legend()
    ax.set_title("EKF SLAM - 실시간 위치 추정")

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=100, interval=100)
plt.show()
