function [xc, yc] = GetBallPos(index)
% GetBallPos - 주어진 인덱스의 이미지에서 공의 중심 좌표를 추정하여 반환
% 입력: index - 처리할 이미지 번호
% 출력: xc, yc - 추정된 공의 x, y 위치 (노이즈 포함)

% 배경 이미지와 초기화 여부를 유지하기 위한 persistent 변수
persistent imgBg
persistent firstRun

% 최초 실행 시 배경 이미지 로딩
if isempty(firstRun)
    imgBg = imread('Img/bg.jpg');  % 기준 배경 이미지 불러오기
    firstRun = 1;
end

% 기본 좌표 초기화 (공을 못 찾을 경우 대비)
xc = 0;
yc = 0;

% 현재 프레임 이미지 불러오기
imgWork = imread(['Img/', int2str(index), '.jpg']); 
imshow(imgWork)  % 현재 이미지 디스플레이

% 배경과 현재 이미지의 차이를 절대값으로 계산
fore = imabsdiff(imgWork, imgBg);

% RGB 각각의 채널에서 차이가 10 이상인 부분을 foreground로 간주
fore = (fore(:,:,1) > 10) | (fore(:,:,2) > 10) | (fore(:,:,3) > 10);

% 논리형 마스크 생성
L = logical(fore);

% 연결된 객체(영역) 분석 - 면적과 중심 좌표 추출
stats = regionprops(L, 'area', 'centroid');

% 각 객체의 면적 배열 추출
area_vector = [stats.Area];

% 가장 큰 영역(즉, 공일 가능성이 높은 영역)의 인덱스 찾기
[~, idx] = max(area_vector);

% 가장 큰 영역의 중심 좌표(Centroid) 추출
centroid = stats(idx(1)).Centroid;

% 노이즈를 포함한 위치 반환 (±15픽셀 정도의 가우시안 노이즈 추가)
xc = centroid(1) + 15*randn;
yc = centroid(2) + 15*randn;
