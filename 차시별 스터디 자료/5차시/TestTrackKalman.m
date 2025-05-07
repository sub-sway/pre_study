clear all  % 모든 변수 초기화

NoOfImg = 24;  % 처리할 이미지 수

% 공의 실제 측정값(xm, ym)과 칼만 필터 추정값(xh, yh)을 저장할 배열 초기화
Xmsaved = zeros(2, NoOfImg);  % 실제 위치 저장용 (2행: x, y)
Xhsaved = zeros(2, NoOfImg);  % 추정 위치 저장용

% 이미지 수만큼 반복
for k = 1:NoOfImg
    % 현재 이미지에서 공의 실제 위치 추정
    [xm, ym] = GetBallPos(k);
    
    % 칼만 필터를 이용해 추정 위치 계산
    [xh, yh] = TrackKalman(xm, ym);
    
    % 현재 프레임의 결과 시각화
    hold on
    plot(xm, ym, 'r*')  % 실제 위치(관측값) - 빨간 별표
    plot(xh, yh, 'bs')  % 칼만 필터 추정 위치 - 파란 사각형

    pause(1)  % 각 프레임 간 1초 대기 (변화 관찰 목적)

    % 결과 저장
    Xmsaved(:, k) = [xm ym]';  % 관측값 저장
    Xhsaved(:, k) = [xh yh]';  % 추정값 저장
end

% 전체 경로 시각화
figure
hold on
plot(Xmsaved(1,:), Xmsaved(2,:), '*')  % 전체 관측 위치 - 별표로 표시
plot(Xhsaved(1,:), Xhsaved(2,:), 's')  % 전체 추정 위치 - 사각형으로 표시
