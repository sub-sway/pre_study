# 수정사항 반영:
# 1. Restart 시 path_line, explored_dots까지 모두 초기화
# 2. waypoint 클릭 시 X표시가 확실히 반영되도록 redraw

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
from matplotlib.widgets import Button

# Grid map
grid = np.array([
    [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
    [1,1,0,1,0,1,1,1,0,1,1,0,1,1,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0,1,0],
    [0,1,0,0,0,1,0,0,0,0,0,1,0,0,0],
    [0,1,0,1,0,1,1,1,1,1,0,1,1,1,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,1,0],
    [1,1,0,1,1,1,1,1,0,1,1,1,0,1,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,1,1,0],
    [0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,1,0,1,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0],
    [1,1,0,1,1,1,1,1,1,1,1,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])

start = (0, 0)
goal = (14, 14)
waypoints = []
moving_obstacle = [7, 7]
obstacle_dir = 1
trace = []
final_path = []
x, y = start
step_size = 0.2
move_index = 1
path_ready = False
trace_index = 0
restart_requested = [False]
path_requested = [False]

directions = [(-1,0), (1,0), (0,-1), (0,1)]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_trace(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, []))
    visited = set()
    trace = []

    while open_set:
        f, cost, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        path = path + [current]
        trace.append(current)
        if current == goal:
            return path, trace
        for dx, dy in directions:
            neighbor = (current[0]+dx, current[1]+dy)
            if (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]
                    and grid[neighbor] == 0 and neighbor not in visited):
                heapq.heappush(open_set, (
                    cost + 1 + heuristic(neighbor, goal),
                    cost + 1,
                    neighbor,
                    path
                ))
    return [], trace

def compute_full_path_with_trace(start, waypoints, goal):
    points = [start] + waypoints + [goal]
    full_path = []
    total_trace = []
    for i in range(len(points)-1):
        sub_path, sub_trace = astar_trace(grid, points[i], points[i+1])
        if not sub_path:
            return [], total_trace
        if i > 0:
            sub_path = sub_path[1:]
        full_path += sub_path
        total_trace += sub_trace
    return full_path, total_trace

# Visualization setup
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.2)
im = ax.imshow(grid, cmap="Greys", origin='upper')
robot_dot, = ax.plot([], [], 'bo', markersize=8, label='Robot')
path_line, = ax.plot([], [], 'k-', lw=2, label='Path')
wp_dots, = ax.plot([], [], 'rx', markersize=6, label='Waypoints')
explored_dots, = ax.plot([], [], 'c.', alpha=0.6, label='Explored')
ax.plot(start[1], start[0], 'go', markersize=8, label='Start')
ax.plot(goal[1], goal[0], 'ro', markersize=8, label='Goal')
ax.legend()
plt.title("A* + Moving Obstacle + Waypoints + Button Start")

# Buttons
reset_ax = plt.axes([0.75, 0.05, 0.2, 0.06])
reset_button = Button(reset_ax, 'Restart', color='lightgray', hovercolor='0.85')
start_ax = plt.axes([0.52, 0.05, 0.2, 0.06])
start_button = Button(start_ax, 'Find Path', color='lightblue', hovercolor='0.85')

def on_reset(event):
    restart_requested[0] = True

def on_start_path(event):
    path_requested[0] = True

reset_button.on_clicked(on_reset)
start_button.on_clicked(on_start_path)

def onclick(event):
    global waypoints
    if event.inaxes == ax and event.button == 1:
        col, row = int(event.xdata + 0.5), int(event.ydata + 0.5)
        if (row, col) not in waypoints and (row, col) != goal and (row, col) != start:
            waypoints.append((row, col))
            wx, wy = zip(*waypoints)
            wp_dots.set_data(wy, wx)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', onclick)

def update(frame):
    global x, y, move_index, final_path, trace, trace_index, path_ready, moving_obstacle, obstacle_dir

    if restart_requested[0]:
        restart_requested[0] = False
        x, y = start
        waypoints.clear()
        wp_dots.set_data([], [])
        path_line.set_data([], [])
        explored_dots.set_data([], [])
        final_path.clear()
        trace.clear()
        move_index = 1
        trace_index = 0
        path_ready = False

    if path_requested[0]:
        path_requested[0] = False
        new_start = (int(round(y)), int(round(x)))
        final_path[:], trace[:] = compute_full_path_with_trace(new_start, waypoints, goal)
        move_index = 1
        trace_index = 0
        path_ready = False

    ox, oy = moving_obstacle
    grid[ox, oy] = 0
    ny = oy + obstacle_dir
    if 0 <= ny < grid.shape[1] and grid[ox, ny] == 0:
        oy = ny
    else:
        obstacle_dir *= -1
        oy = oy + obstacle_dir if 0 <= oy + obstacle_dir < grid.shape[1] else oy
    moving_obstacle = [ox, oy]
    grid[ox, oy] = 1
    im.set_data(grid)

    if not path_ready:
        if trace_index < len(trace):
            px, py = zip(*trace[:trace_index+1])
            explored_dots.set_data(py, px)
            trace_index += 1
            return [explored_dots]
        else:
            if final_path:
                fx, fy = zip(*final_path)
                path_line.set_data(fy, fx)
                path_ready = True
            return [explored_dots, path_line]

    if not final_path or move_index >= len(final_path):
        return [robot_dot, path_line, explored_dots, wp_dots]

    tx, ty = final_path[move_index]
    dx, dy = tx - x, ty - y
    dist = np.hypot(dx, dy)
    if dist < step_size:
        x, y = tx, ty
        move_index += 1
    else:
        x += step_size * dx / dist
        y += step_size * dy / dist

    robot_dot.set_data([y], [x])
    return [robot_dot, path_line, explored_dots, wp_dots]

ani = animation.FuncAnimation(fig, update, frames=2000, interval=20, blit=True, repeat=False)
plt.show()
