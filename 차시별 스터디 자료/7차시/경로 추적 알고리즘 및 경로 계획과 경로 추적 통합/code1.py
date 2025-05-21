import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
from matplotlib.animation import PillowWriter

# 복잡한 맵
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

# A* 관련 변수
open_set = []
heapq.heappush(open_set, (0, 0, start, []))
visited = set()
path_found = False
final_path = []

directions = [(-1,0), (1,0), (0,-1), (0,1)]
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# 로봇 이동 변수
x, y = start
step_size = 0.1
move_index = [1]

# 시각화 설정
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(grid, cmap="Greys", origin='upper')
ax.plot(start[1], start[0], 'go', markersize=8, label='Start')
ax.plot(goal[1], goal[0], 'ro', markersize=8, label='Goal')

explored_x, explored_y = [], []
explored_dots, = ax.plot([], [], 'c.', alpha=0.6, label='Exploring')
path_line, = ax.plot([], [], 'k-', lw=2, label='Final Path')
robot_dot, = ax.plot([], [], 'bo', markersize=8, label='Robot')

ax.legend()
plt.title("Real-time A* Search + Path Lock + Point Tracking")

# 업데이트 함수
def update(frame):
    global x, y, path_found, final_path

    if not path_found:
        if not open_set:
            print("경로 없음")
            return [explored_dots, path_line, robot_dot]

        f, cost, current, path = heapq.heappop(open_set)
        if current in visited:
            return [explored_dots, path_line, robot_dot]
        visited.add(current)
        explored_x.append(current[1])
        explored_y.append(current[0])
        explored_dots.set_data(explored_x, explored_y)

        path = path + [current]
        if current == goal:
            path_found = True
            final_path = path
            fx, fy = zip(*final_path)
            path_line.set_data(fy, fx)
            return [explored_dots, path_line, robot_dot]

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < grid.shape[0] and
                0 <= neighbor[1] < grid.shape[1] and
                grid[neighbor] == 0 and neighbor not in visited):
                heapq.heappush(open_set, (
                    cost + 1 + heuristic(neighbor, goal),
                    cost + 1,
                    neighbor,
                    path
                ))
        return [explored_dots, path_line, robot_dot]

    # 경로 추적 단계
    if move_index[0] >= len(final_path):
        return [robot_dot, path_line]

    tx, ty = final_path[move_index[0]]
    dx, dy = tx - x, ty - y
    dist = np.hypot(dx, dy)
    if dist < step_size:
        x, y = tx, ty
        move_index[0] += 1
    else:
        x += step_size * dx / dist
        y += step_size * dy / dist

    robot_dot.set_data([y], [x])
    return [robot_dot, path_line]

# 애니메이션 생성
ani = animation.FuncAnimation(fig, update, frames=1000, interval=30, blit=True, repeat=False)

# GIF 저장
writer = PillowWriter(fps=30)
ani.save("astar_path.gif", writer=writer)

# 시각화 (옵션)
plt.show()
