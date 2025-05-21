import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# 시뮬레이션 설정
dt = 0.1
T = 20
steps = int(T / dt)
time = np.arange(0, T, dt)

# 단위 계단 입력
def unit_step(t): return 1.0

# 제어기 설정
controllers = {
    "P": {"color": "r"},
    "PI": {"color": "g"},
    "PID": {"color": "b"}
}

# 초기 PID 계수
initial_params = {
    "P": {"Kp": 2.0, "Ki": 0.0, "Kd": 0.0},
    "PI": {"Kp": 2.0, "Ki": 3.0, "Kd": 0.0},
    "PID": {"Kp": 2.0, "Ki": 3.0, "Kd": 0.3}
}

# 출력 변수 초기화
for name in controllers:
    controllers[name].update({
        **initial_params[name],
        "y": 0.0, "integral": 0.0, "prev_error": 0.0,
        "xdata": [], "ydata": [],
        "max_y": 0.0, "overshoot": 0.0
    })

# Figure 및 레이아웃 설정
fig = plt.figure(figsize=(14, 6))
ax = fig.add_axes([0.05, 0.1, 0.6, 0.8])  # 그래프는 왼쪽에 위치

# 실시간 출력 라인 및 텍스트 박스
lines = {}
texts = {}
for i, (name, ctrl) in enumerate(controllers.items()):
    line, = ax.plot([], [], ctrl["color"], label=f"{name} 제어")
    lines[name] = line
    texts[name] = ax.text(0.02, 2.2 - 0.2 * i, "", color=ctrl["color"],
                          fontsize=9, transform=ax.transAxes,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.plot(time, [1.0] * len(time), 'k--', label="입력 (Step)")
ax.set_xlim(0, T)
ax.set_ylim(0, 2.5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Output")
ax.set_title("Step Response with Overshoot (슬라이더 우측 배치)")
ax.grid(True)
ax.legend()

# 슬라이더 구성 (오른쪽에 세로 배치)
sliders = {}
slider_width = 0.15
slider_height = 0.03
padding = 0.005
for i, name in enumerate(controllers.keys()):
    for j, param in enumerate(["Kp", "Ki", "Kd"]):
        # 슬라이더 위치 계산 (세로 정렬)
        left = 0.7
        bottom = 0.85 - (i * 3 + j) * (slider_height + padding)
        ax_slider = fig.add_axes([left, bottom, slider_width, slider_height])
        slider = Slider(ax_slider, f"{name} {param}", 0.0, 10.0,
                        valinit=initial_params[name][param])
        sliders[f"{name}_{param}"] = slider

# 초기화 함수
def reset_controllers():
    for name, ctrl in controllers.items():
        ctrl["y"] = 0.0
        ctrl["integral"] = 0.0
        ctrl["prev_error"] = 0.0
        ctrl["xdata"] = []
        ctrl["ydata"] = []
        ctrl["max_y"] = 0.0
        ctrl["overshoot"] = 0.0

def init():
    reset_controllers()
    for name in controllers:
        lines[name].set_data([], [])
        texts[name].set_text("")
    return list(lines.values()) + list(texts.values())

# 업데이트 함수
def update(frame):
    t = time[frame]
    ref = unit_step(t)

    for name, ctrl in controllers.items():
        ctrl["Kp"] = sliders[f"{name}_Kp"].val
        ctrl["Ki"] = sliders[f"{name}_Ki"].val
        ctrl["Kd"] = sliders[f"{name}_Kd"].val

        error = ref - ctrl["y"]
        ctrl["integral"] += error * dt
        derivative = (error - ctrl["prev_error"]) / dt
        u = ctrl["Kp"] * error + ctrl["Ki"] * ctrl["integral"] + ctrl["Kd"] * derivative
        ctrl["prev_error"] = error

        y_dot = -ctrl["y"] + u
        ctrl["y"] += y_dot * dt

        ctrl["xdata"].append(t)
        ctrl["ydata"].append(ctrl["y"])

        if ctrl["y"] > ctrl["max_y"]:
            ctrl["max_y"] = ctrl["y"]
        if ctrl["max_y"] > 1.0:
            ctrl["overshoot"] = (ctrl["max_y"] - 1.0) * 100

        lines[name].set_data(ctrl["xdata"], ctrl["ydata"])
        texts[name].set_text(f"{name} OS: {ctrl['overshoot']:.1f}%")

    return list(lines.values()) + list(texts.values())

ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=100)
plt.show()
