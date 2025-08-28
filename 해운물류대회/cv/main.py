# streamlit run main.py
import streamlit as st
import cv2
import time
import threading
import os
import tempfile
import torch
from ultralytics import YOLO

# ---------------------------
# 페이지/세션 초기 설정
# ---------------------------
st.set_page_config(page_title="HiVis + Fire 모니터링", layout="wide")
st.title("🟢 HiVis + Fire YOLO 실시간 모니터링 (Streamlit)")

if "running" not in st.session_state:
    st.session_state.running = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "hivis_model" not in st.session_state:
    st.session_state.hivis_model = None
if "fire_model" not in st.session_state:
    st.session_state.fire_model = None
if "last_beep_hivis" not in st.session_state:
    st.session_state.last_beep_hivis = 0.0
if "last_beep_fire" not in st.session_state:
    st.session_state.last_beep_fire = 0.0
if "last_frame_time" not in st.session_state:
    st.session_state.last_frame_time = None

# ---------------------------
# 유틸 함수
# ---------------------------
def is_inside(inner, outer, eps: float = 0.0):
    """
    inner 박스(x1,y1,x2,y2)의 네 변 중 최소 3변이 outer 경계 안에 있으면 True.
    eps는 부동소수 오차 허용치.
    """
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    conds = [
        x1 >= ox1 - eps,
        y1 >= oy1 - eps,
        x2 <= ox2 + eps,
        y2 <= oy2 + eps,
    ]
    return sum(conds) >= 3

def beep_async(freq, dur):
    # 로컬(윈도우)에서만 작동하는 비프. 브라우저 스피커로는 직접 못 냅니다.
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()
    except Exception:
        pass

def save_temp(uploaded_file):
    if uploaded_file is None:
        return None
    import time as _t
    tpath = os.path.join(tempfile.gettempdir(), f"upl_{int(_t.time())}_{uploaded_file.name}")
    with open(tpath, "wb") as f:
        f.write(uploaded_file.read())
    return tpath

# ---------------------------
# 사이드바 (설정)
# ---------------------------
with st.sidebar:
    st.header("설정")

    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    device = st.selectbox("장치(Device)", [device_default, "cpu"], index=0)

    conf_hivis = st.slider("HiVis/Person conf", 0.05, 0.95, 0.50, 0.05)
    conf_fire  = st.slider("Fire/Smoke conf", 0.05, 0.95, 0.50, 0.05)
    iou_thres  = st.slider("NMS IoU", 0.1, 0.9, 0.45, 0.05)
    imgsz      = st.select_slider("입력 해상도", options=[320, 384, 448, 512, 576, 640, 704, 768], value=640)
    show_fps   = st.checkbox("FPS 표시", value=True)

    # 팝업/비프 디바운싱 간격
    beep_interval_s = st.number_input("Beep/팝업 최소 간격(초)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

    # ✅ 팝업 on/off 옵션
    show_fire_popup = st.checkbox("화재 감지 시 팝업 띄우기", value=True)
    show_hivis_popup = st.checkbox("사람 HiVis 미착용 시 팝업 띄우기", value=True)

    st.divider()
    st.caption("모델 경로/업로드")
    default_hivis = st.text_input("HiVis 모델 경로", value="C:\\code\\pro\\YOLO-HiVis\\models\\HiVisModel.pt")
    default_fire  = st.text_input("Fire 모델 경로",  value="C:\\code\\pro\\wildfire-detection\\fire-models\\fire_s.pt")

    up_hivis = st.file_uploader("HiVis 모델 업로드(선택)", type=["pt", "onnx"], key="up_hivis")
    up_fire  = st.file_uploader("Fire  모델 업로드(선택)", type=["pt", "onnx"], key="up_fire")

    hivis_path = save_temp(up_hivis) or default_hivis
    fire_path  = save_temp(up_fire)  or default_fire

    st.divider()
    source = st.radio("입력 소스", ["웹캠", "RTSP/HTTP URL"], index=0)
    if source == "웹캠":
        cam_index = st.number_input("웹캠 인덱스", min_value=0, max_value=10, value=0, step=1)
        use_dshow = st.checkbox("Windows DSHOW 사용", value=True, help="일부 웹캠에서 지연/호환성 개선")
    else:
        rtsp_url = st.text_input("RTSP/HTTP URL", placeholder="rtsp://... 또는 http(s)://...")

    col_a, col_b = st.columns(2)
    with col_a:
        start_btn = st.button("▶ 실행", type="primary")
    with col_b:
        stop_btn = st.button("⏹ 중지")

# ---------------------------
# 시작/중지 버튼 로직
# ---------------------------
if start_btn:
    # 1) 모델 로드
    try:
        st.session_state.hivis_model = YOLO(hivis_path)
        st.session_state.fire_model  = YOLO(fire_path)
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.stop()

    # 2) 입력 소스 열기
    if source == "웹캠":
        if use_dshow:
            cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(int(cam_index))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        if not rtsp_url:
            st.warning("URL을 입력해 주세요.")
            st.stop()
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        st.error("비디오 소스를 열 수 없습니다.")
        st.stop()

    # 3) 상태 초기화
    st.session_state.cap = cap
    st.session_state.running = True
    st.session_state.last_frame_time = None
    st.session_state.last_beep_hivis = 0.0
    st.session_state.last_beep_fire  = 0.0

if stop_btn and st.session_state.running:
    st.session_state.running = False
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

# ---------------------------
# 표시 영역
# ---------------------------
video_col, info_col = st.columns([4, 1], vertical_alignment="top")
frame_placeholder = video_col.empty()
info_placeholder  = info_col.container()

# ---------------------------
# 메인 루프
# ---------------------------
while st.session_state.running and st.session_state.cap is not None:
    ok, frame = st.session_state.cap.read()
    if not ok:
        st.warning("프레임을 읽지 못했습니다. (스트림 종료/카메라 문제)")
        st.session_state.running = False
        break

    # YOLO 추론 (두 모델)
    hivis_res = st.session_state.hivis_model.predict(
        source=frame, conf=conf_hivis, iou=iou_thres, imgsz=imgsz, device=device, verbose=False
    )
    fire_res = st.session_state.fire_model.predict(
        source=frame, conf=conf_fire,  iou=iou_thres, imgsz=imgsz, device=device, verbose=False
    )

    hivis_names = hivis_res[0].names
    fire_names  = fire_res[0].names

    person_boxes, hivis_boxes = [], []

    # HiVis/person 박스 파싱 및 표시
    for box in hivis_res[0].boxes:
        cls_id = int(box.cls)
        cls_name = hivis_names.get(cls_id, str(cls_id)) if isinstance(hivis_names, dict) else str(cls_id)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if cls_name == "person":
            person_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        elif cls_name in ["hivis", "hi-vis"]:
            hivis_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,0), 2)

    # 사람 박스 내 HiVis 체크 + 알람(디바운싱) + ✅ 외부인 팝업
    now = time.time()
    for p in person_boxes:
        hivis_inside = any(is_inside(h, p) for h in hivis_boxes)
        if not hivis_inside and (now - st.session_state.last_beep_hivis) > beep_interval_s:
            st.session_state.last_beep_hivis = now
            beep_async(1000, 300)
            if show_hivis_popup:
                st.toast("외부인 감지", icon="🚨")  # ← 여기서 팝업

    # 화재/연기 박스 및 알람 (+ 팝업)
    n_fire = 0
    for box in fire_res[0].boxes:
        cls_id = int(box.cls)
        cls_name = fire_names.get(cls_id, str(cls_id)) if isinstance(fire_names, dict) else str(cls_id)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if cls_name in ["fire", "smoke"]:
            n_fire += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

            if (now - st.session_state.last_beep_fire) > beep_interval_s:
                st.session_state.last_beep_fire = now
                st.write(f"[화재 감지] 위치: {[int(x1), int(y1), int(x2), int(y2)]}")
                beep_async(2000, 300)
                if show_fire_popup:
                    st.toast("화재가 감지되었습니다.", icon="🔥")

    # 주석 프레임 표시 (BGR->RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    # 정보 패널
    with info_placeholder:
        st.markdown("**Detection Info**")
        rows = [
            f"- person: {len(person_boxes)}",
            f"- hivis : {len(hivis_boxes)}",
            f"- fire/smoke: {n_fire}",
        ]
        st.write("\n".join(rows))

        # FPS
        if show_fps:
            nowt = time.time()
            if st.session_state.last_frame_time is None:
                fps = 0.0
            else:
                dt = nowt - st.session_state.last_frame_time
                fps = 1.0 / dt if dt > 0 else 0.0
            st.session_state.last_frame_time = nowt
            st.caption(f"**{fps:.1f} FPS**")

    # UI 갱신 여유 (CPU 사용량 방지)
    time.sleep(0.001)

# 종료 처리
if not st.session_state.running and st.session_state.cap is not None:
    try:
        st.session_state.cap.release()
    except Exception:
        pass
    st.session_state.cap = None

st.caption("팁: 속도가 느리면 입력 해상도 낮추기, 경량 모델(YOLOn/s) 사용, conf 상향을 시도하세요. CUDA 가능 시 'cuda' 선택.")
