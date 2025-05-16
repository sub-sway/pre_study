import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import numpy as np

# ---------- 입력 맵 ----------
matrix = [
    [True, True, True, False, False, False, False],
    [True, False, True, False, False, False, False],
    [True, False, True, True, True, True, True],
    [True, False, True, False, False, False, True],
    [True, False, True, False, True, True, True],
    [True, False, True, False, True, False, False],
    [True, True, True, True, True, True, True],
]
start = (2, 2)
goal = (6, 6)

rows, cols = len(matrix), len(matrix[0])

# ---------- 휴리스틱 계산 ----------
h_map = np.zeros((rows, cols), dtype=int)
for r in range(rows):
    for c in range(cols):
        h_map[r][c] = abs(r - goal[0]) + abs(c - goal[1]) if matrix[r][c] else -1

# ---------- 프레임 저장용 ----------
frames = []

# ---------- A* 알고리즘 ----------
def a_star_grid_path():
    came_from = {}
    g_score = {start: 0}
    f_score = {start: h_map[start[0]][start[1]]}
    open_set = [(f_score[start], start)]
    open_set_tracker = {start}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_tracker.discard(current)
        visited.add(current)

        # 현재 상태 저장
        frame = np.full((rows, cols), '', dtype=object)
        for r in range(rows):
            for c in range(cols):
                if not matrix[r][c]:
                    frame[r][c] = ''  # 장애물
                elif (r, c) == start:
                    frame[r][c] = 'S'
                elif (r, c) == goal:
                    frame[r][c] = 'G'
                elif (r, c) == current:
                    frame[r][c] = '●'
                elif (r, c) in open_set_tracker:
                    frame[r][c] = 'O'
                elif (r, c) in visited:
                    frame[r][c] = str(h_map[r][c])
                else:
                    frame[r][c] = ''
        frames.append(frame)

        if current == goal:
            # 경로 추적
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            for i in range(1, len(path) + 1):
                frame = np.full((rows, cols), '', dtype=object)
                for r in range(rows):
                    for c in range(cols):
                        if not matrix[r][c]:
                            frame[r][c] = ''
                        elif (r, c) == start:
                            frame[r][c] = 'S'
                        elif (r, c) == goal:
                            frame[r][c] = 'G'
                        elif (r, c) in path[:i]:
                            frame[r][c] = '●'
                        elif (r, c) in visited:
                            frame[r][c] = str(h_map[r][c])
                        else:
                            frame[r][c] = ''
                frames.append(frame)
            break

        # 이웃 노드 확인
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc]:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + h_map[nr][nc]
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_tracker.add(neighbor)

# ---------- 시각화 함수 ----------
def update(frame):
    ax.clear()
    ax.axis('off')
    table = ax.table(cellText=frame.tolist(), loc='center', cellLoc='center', edges='closed')
    table.scale(1.2, 1.5)

# ---------- 실행 ----------
fig, ax = plt.subplots(figsize=(7, 6))
a_star_grid_path()
ani = animation.FuncAnimation(fig, update, frames=frames, interval=500, repeat=False)
plt.show()
