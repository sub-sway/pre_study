import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import math

goal = np.array([10.0, 10.0])
start = np.array([0.0, 0.0, 0.0])
obstacles = np.array([[3, 3], [6, 5], [7, 8], [5, 7]])
max_speed = 1.0
min_speed = -0.5
max_yawrate = np.deg2rad(40)
v_reso = 0.1
yawrate_reso = 0.1
dt = 0.1
predict_time = 1.0
robot_radius = 0.5
threshold = 0.5

state = start.copy()
dwa_path = [state[:2].copy()]

def motion(x, u, dt):
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt
    return x

def calc_trajectory(x_init, v, y, time):
    x = x_init.copy()
    traj = [x[:2].copy()]
    for _ in np.arange(0, time, dt):
        x = motion(x, [v, y], dt)
        traj.append(x[:2].copy())
    return np.array(traj)

def dwa_control(x, goal, obstacles):
    best_u = [0.0, 0.0]
    min_cost = float('inf')
    for v in np.arange(min_speed, max_speed, v_reso):
        for y in np.arange(-max_yawrate, max_yawrate, yawrate_reso):
            traj = calc_trajectory(x.copy(), v, y, predict_time)
            dist = np.linalg.norm(traj[-1] - goal)
            collision = False
            for obs in obstacles:
                if np.any(np.linalg.norm(traj - obs, axis=1) <= robot_radius + 0.2):
                    collision = True
                    break
            if not collision and dist < min_cost:
                min_cost = dist
                best_u = [v, y]
    return best_u


fig, ax = plt.subplots(figsize=(6, 6))
line, = ax.plot([], [], 'b-', label='Path (DWA)')
robot_dot, = ax.plot([], [], 'bo', markersize=6)
goal_dot = ax.plot(goal[0], goal[1], 'ro', label='Goal')[0]
start_dot = ax.plot(start[0], start[1], 'go', label='Start')[0]
for obs in obstacles:
    ax.add_patch(Circle(obs, 0.5, color='gray'))

ax.set_xlim(-1, 12)
ax.set_ylim(-1, 12)
ax.set_title("DWA Path Planning")
ax.legend()
ax.grid(True)
ax.set_aspect('equal')

def update(frame):
    global state
    if np.linalg.norm(state[:2] - goal) < threshold:
        return line, robot_dot
    u = dwa_control(state, goal, obstacles)
    state = motion(state, u, dt)
    dwa_path.append(state[:2].copy())
    path_np = np.array(dwa_path)
    line.set_data(path_np[:, 0], path_np[:, 1])
    robot_dot.set_data([state[0]], [state[1]])
    return line, robot_dot

ani = FuncAnimation(fig, update, frames=300, interval=100, blit=False)
plt.show()
