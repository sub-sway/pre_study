import cv2
import numpy as np

# 기준 이미지 로딩 및 크기 조정
reference = cv2.imread('ref.jpg', 0)
if reference is None:
    print("기준 이미지(gwajam.jpg)를 불러올 수 없습니다.")
    exit()

# 기준 크기 설정
STANDARD_WIDTH, STANDARD_HEIGHT = 640, 480
reference = cv2.resize(reference, (STANDARD_WIDTH, STANDARD_HEIGHT))

# ORB 및 매칭기 초기화
orb = cv2.ORB_create(1000)
kp2, des2 = orb.detectAndCompute(reference, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# 웹캠 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (STANDARD_WIDTH, STANDARD_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray, None)

    match_img = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH * 2, 3), dtype=np.uint8)

    if des1 is not None and des2 is not None:
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist() if mask is not None else None

            match_img = cv2.drawMatches(
                gray, kp1, reference, kp2, good, None,
                matchesMask=matches_mask,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            match_img = cv2.resize(match_img, (STANDARD_WIDTH * 2, STANDARD_HEIGHT))

    cv2.imshow('Live Camera', frame)
    cv2.imshow('RANSAC Matching', match_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
