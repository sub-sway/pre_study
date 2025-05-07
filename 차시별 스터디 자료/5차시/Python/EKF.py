import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 설정
dt = 0.1
Q = np.diag([0.01, 0.01, np.deg2rad(1.0)])**2         # 이동 잡음
R = np.diag([0.5, np.deg2rad(10.0)])**2               # 센서 잡음 (거리, 각도)
landmarks = np.array([[5.0, 10.0], [10.0, 5.0], [15.0, 15.0]])  # 랜드마크 3개

# 초기 상태 [x, y, theta]
x_true = np.array([0.0, 0.0, 0.0])
x_est = np.array([0.0, 0.0, 0.0])
P = np.eye(3)

# 경로 저장
true_path = []
est_path = []

# 이동 모델
def motion_model(x, u):
    theta = x[2]
    return np.array([
        x[0] + u[0] * np.cos(theta) * dt,
        x[1] + u[0] * np.sin(theta) * dt,
        x[2] + u[1] * dt
    ])

# 이동 자코비안
def jacob_F(x, u):
    theta = x[2]
    return np.array([
        [1, 0, -u[0] * np.sin(theta) * dt],
        [0, 1,  u[0] * np.cos(theta) * dt],
        [0, 0, 1]
    ])

# 측정 모델 (거리, 각도)
def observation_model(x, lm):
    dx = lm[0] - x[0]
    dy = lm[1] - x[1]
    return np.array([
        np.sqrt(dx**2 + dy**2),
        np.arctan2(dy, dx) - x[2]
    ])

# 측정 자코비안
def jacob_H(x, lm):
    dx = lm[0] - x[0]
    dy = lm[1] - x[1]
    q = dx**2 + dy**2
    sqrt_q = np.sqrt(q)
    return np.array([
        [-dx / sqrt_q, -dy / sqrt_q, 0],
        [dy / q, -dx / q, -1]
    ])

# EKF 알고리즘
def ekf(x_est, P, z, u, lm):
    x_pred = motion_model(x_est, u)
    F = jacob_F(x_est, u)
    P_pred = F @ P @ F.T + Q

    z_pred = observation_model(x_pred, lm)
    H = jacob_H(x_pred, lm)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    y = z - z_pred
    y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi  # 각도 정규화
    x_upd = x_pred + K @ y
    P_upd = (np.eye(len(x_est)) - K @ H) @ P_pred

    return x_upd, P_upd

# 시뮬레이션 1회
def run_simulation():
    global x_true, x_est, P

    u = np.array([1.0, np.deg2rad(20)])  # 선속도, 각속도
    x_true[:] = motion_model(x_true, u)

    for lm in landmarks:
        z = observation_model(x_true, lm) + np.random.multivariate_normal([0, 0], R)
        x_est[:], P[:] = ekf(x_est, P, z, u, lm)

    true_path.append(x_true[:2])
    est_path.append(x_est[:2])

# 애니메이션 갱신
fig, ax = plt.subplots()

def update(frame):
    run_simulation()

    ax.clear()
    ax.set_xlim(-5, 20)
    ax.set_ylim(-5, 20)
    ax.set_title("2D EKF SLAM Real-Time Simulation")
    ax.plot(*zip(*true_path), 'b-', label="True Path")
    ax.plot(*zip(*est_path), 'r--', label="EKF Estimate")
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='g', marker='x', label="Landmarks")
    ax.legend()
    ax.grid()

# 애니메이션 객체 유지 필요
ani = FuncAnimation(fig, update, frames=200, interval=100)
plt.show()
