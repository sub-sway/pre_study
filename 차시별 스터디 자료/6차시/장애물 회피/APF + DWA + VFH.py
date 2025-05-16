# A* + APF, DWA, VFH 비교 시뮬레이션
# APF만 장애물 2개 사용: (2,7), (10,20), DWA/VFH는 기존 장애물 모두 사용

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import heapq
import math

# 환경 설정
grid_size = (21, 21)

# 모든 장애물 (DWA, VFH용)
all_obstacles = [
    (0, 5), (0, 10), (0, 15), (0, 20),
    (5, 20), (10, 20), (15, 20),
    (1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
    (6, 15), (7, 16), (8, 17), (9, 18), (10, 19)
]
# APF는 이 중 일부만 사용
apf_obstacles = [(7, 5), (20, 10)]

# 지도 생성
obstacles_map = np.zeros(grid_size)
for x, y in all_obstacles:
    obstacles_map[y, x] = 1
obstacles_all = np.argwhere(obstacles_map == 1) + 0.5
obstacles_apf = np.array([[y + 0.5, x + 0.5] for x, y in apf_obstacles])

start = (1, 1)
goal = (20, 20)

# A* 알고리즘
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(start, goal, grid):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()
    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[1] and 0 <= neighbor[1] < grid.shape[0] and grid[neighbor[1], neighbor[0]] == 0:
                heapq.heappush(open_set, (cost + 1 + heuristic(neighbor, goal), cost + 1, neighbor, path + [neighbor]))
    return []

path_astar = a_star(start, goal, obstacles_map)
path_points = np.array([[x + 0.5, y + 0.5] for x, y in path_astar])

# 초기 상태
q_apf = np.array([start[0] + 0.5, start[1] + 0.5])
state_dwa = np.array([start[0] + 0.5, start[1] + 0.5, 0.0])
robot_vfh = np.array([start[0] + 0.5, start[1] + 0.5])

# 경로 기록
apf_path, dwa_path, vfh_path = [q_apf.copy()], [state_dwa[:2].copy()], [robot_vfh.copy()]

# DWA 설정
max_speed = 0.5
min_speed = 0.0
max_yawrate = np.deg2rad(40)
v_reso = 0.1
yawrate_reso = 0.1
dt = 0.1
predict_time = 1.0
robot_radius = 0.5

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

def dwa_control(x, path_points, obstacles):
    best_u = [0.0, 0.0]
    min_cost = float('inf')
    closest_idx = np.argmin(np.linalg.norm(path_points - x[:2], axis=1))
    goal_point = path_points[min(closest_idx + 1, len(path_points) - 1)]
    for v in np.arange(min_speed, max_speed + v_reso, v_reso):
        for y in np.arange(-max_yawrate, max_yawrate + yawrate_reso, yawrate_reso):
            traj = calc_trajectory(x.copy(), v, y, predict_time)
            dist = np.linalg.norm(traj[-1] - goal_point)
            collision = False
            for obs in obstacles:
                if np.any(np.linalg.norm(traj - obs[::-1], axis=1) <= robot_radius + 0.2):
                    collision = True
                    break
            if not collision and dist < min_cost:
                min_cost = dist
                best_u = [v, y]
    return best_u

# APF 설정
eta, zeta, d0, step_size = 100, 1, 2.0, 0.1

def gradient_potential(q, path_points, obstacles):
    closest_idx = np.argmin(np.linalg.norm(path_points - q, axis=1))
    goal_point = path_points[min(closest_idx + 1, len(path_points) - 1)]
    grad = zeta * (q - goal_point)
    for obs in obstacles:
        diff = q - obs
        d = np.linalg.norm(diff)
        if 0.3 < d <= d0:
            grad += eta * (1/d - 1/d0) * (1/d**3) * diff
    return grad


# VFH 설정
sensor_range = 3.0
angular_sectors = 72

def vfh_steering_direction(robot_pos, path_points, obstacles):
    angles = np.linspace(-np.pi, np.pi, angular_sectors)
    density = np.zeros_like(angles)

    for obs in obstacles:
        rel = obs[::-1] - robot_pos
        dist = np.linalg.norm(rel)
        if dist == 0 or dist > sensor_range:
            continue
        influence = max(0, sensor_range - dist) / sensor_range
        idx = np.argmin(np.abs(angles - np.arctan2(rel[1], rel[0])))
        density[idx] += influence * 3

    mask = density > 0.4
    closest_idx = np.argmin(np.linalg.norm(path_points - robot_pos, axis=1))
    target_point = path_points[min(closest_idx + 1, len(path_points) - 1)]
    goal_angle = np.arctan2(target_point[1] - robot_pos[1], target_point[0] - robot_pos[0])
    open_idx = np.where(~mask)[0]
    if len(open_idx) == 0:
        return None
    best_idx = open_idx[np.argmin(np.abs(angles[open_idx] - goal_angle))]
    return angles[best_idx]

# 시각화
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
titles = ["A* + APF (2 Obstacles)", "A* + DWA", "A* + VFH"]
colors = ['g', 'b', 'm']
paths = [apf_path, dwa_path, vfh_path]
dots, lines = [], []

for i, ax in enumerate(axs):
    ax.set_xlim(-1, grid_size[0])
    ax.set_ylim(-1, grid_size[1])
    ax.set_title(titles[i])
    ax.grid(True)
    ax.set_aspect('equal')
    if i == 0:
        for obs in obstacles_apf:
            ax.add_patch(Circle((obs[0], obs[1]), 0.4, color='gray'))
    else:
        for obs in obstacles_all:
            ax.add_patch(Circle((obs[1], obs[0]), 0.4, color='gray'))
    ax.plot(goal[0]+0.5, goal[1]+0.5, 'ro')
    ax.plot(start[0]+0.5, start[1]+0.5, 'go')
    line, = ax.plot([], [], colors[i]+'-', label=titles[i])
    dot, = ax.plot([], [], colors[i]+'o', markersize=6)
    lines.append(line)
    dots.append(dot)

def update(frame):
    global q_apf, state_dwa, robot_vfh
    # APF
    grad = gradient_potential(q_apf, path_points, obstacles_apf)
    if np.linalg.norm(grad) != 0:
        q_apf -= step_size * grad / np.linalg.norm(grad)
    apf_path.append(q_apf.copy())
    # DWA
    u = dwa_control(state_dwa, path_points, obstacles_all)
    state_dwa = motion(state_dwa, u, dt)
    dwa_path.append(state_dwa[:2].copy())
    # VFH
    angle = vfh_steering_direction(robot_vfh, path_points, obstacles_all)
    if angle is not None:
        robot_vfh += step_size * np.array([np.cos(angle), np.sin(angle)])
    vfh_path.append(robot_vfh.copy())

    for path, line, dot in zip([apf_path, dwa_path, vfh_path], lines, dots):
        path_np = np.array(path)
        line.set_data(path_np[:, 0], path_np[:, 1])
        dot.set_data([path_np[-1, 0]], [path_np[-1, 1]])
    return lines + dots

ani = FuncAnimation(fig, update, frames=500, interval=100, blit=False)
plt.show()
