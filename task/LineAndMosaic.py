import cv2

# 모자이크 처리 함수
def mosaic(img, msize):
    # 이미지 크기만큼 픽셀 탐색
    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            xpoint = int(x / msize) * msize
            ypoint = int(y / msize) * msize  # 모자이크 처리할 픽셀 설정
            mainColor = img[ypoint][xpoint]  # 모자이크 처리할 RGB 설정
            img[y][x] = mainColor  # 해당하는 좌표의 픽셀을 저장한 mainColor RGB로 대체
    return img  # 모자이크 처리된 이미지 반환


def line(img, point1, point2):
    # point1, poin2 좌표에 좌표표시
    cv2.putText(img, str(point1), (point1[0], point1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, str(point2), (point2[0], point2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 수직선 또는 수평선일경우는 따로 처리

    # 수직선일경우
    if point2[0] - point1[0] == 0:
        if point2[1] < point1[1]:
            for y in range(point2[1], point1[1]):  # 수직선 길이만큼 y좌표 순환
                img[y][point2[0]] = [0, 0, 255]  # 해당좌표에 색칠
        else:
            for y in range(point1[1], point2[1]):  # 수직선 길이만큼 y좌표 순환
                img[y][point2[0]] = [0, 0, 255]  # 해당좌표에 색칠
    # 수평선일 경우
    elif point2[1] - point1[1] == 0:
        if point2[0] < point1[0]:
            for x in range(point2[0], point1[0]):  # 수평선 길이만큼 x좌표 순환
                img[point2[1]][x] = [0, 0, 255]  # 해당좌표에 색칠
        else:
            for x in range(point1[0], point2[0]):  # 수평선 길이만큼 x좌표 순환
                img[point2[1]][x] = [0, 0, 255]  # 해당좌표에 색칠

    # 수직, 수평선이 아닐 경우
    else:
        # 각도 계산
        dy = (point2[1] - point1[1]) / (point2[0] - point1[0])
        dx = (point2[0] - point1[0]) / (point2[1] - point1[1])

        # point1와 point2 직선의 가로길이와 세로길이 중 더 긴 길이 선별 후 더 긴 길이를 기준으로 실행

        # 가로가 더 긴 경우
        if abs(point1[0] - point2[0]) > abs(point1[1] - point2[1]):
            if point2[0] < point1[0]:
                sy = point2[0]
                for x in range(point2[0], point1[0]):  # point1과 poin2사이의 가로길이 탐색
                    img[int(sy)][x] = [0, 0, 255]  # 해당좌표에 색칠
                    sy = sy + dy  # 기울기만큼 y좌표 수정
            else:
                sy = point1[0]
                for x in range(point1[0], point2[0]):  # point1과 poin2사이의 가로길이 탐색
                    img[int(sy)][x] = [0, 0, 255]  # 해당좌표에 색칠
                    sy = sy + dy  # 기울기만큼 y좌표 수정
        # 세로가 더 긴 경우
        else:
            if point2[1] < point1[1]:
                sx = point2[0]
                for y in range(point2[1], point1[1]):  # point1과 poin2사이의 세로길이 탐색
                    img[y][int(sx)] = [0, 0, 255]  # 해당좌표에 색칠
                    sx = sx + dx  # 기울기만큼 x좌표 수정
            else:
                sx = point1[1]
                for y in range(point1[1], point2[1]):  # point1과 poin2사이의 세로길이 탐색
                    img[y][int(sx)] = [0, 0, 255]  # 해당좌표에 색칠
                    sx = sx + dx  # 기울기만큼 x좌표 수정

    return img  # 이미지 반환


# 이하는 테스트용 코드
x = 150
y = 150


def imshow():
    img = cv2.imread('img/test.jpg')
    img2 = line(img, [x, y], [50, 50])
    img2 = line(img, [x, y], [50, 150])
    img2 = line(img, [x, y], [50, 250])
    img2 = line(img, [x, y], [150, 50])
    img2 = line(img, [x, y], [150, 250])
    img2 = line(img, [x, y], [250, 50])
    img2 = line(img, [x, y], [250, 150])
    img2 = line(img, [x, y], [250, 250])
    cv2.imshow('line', img2)
    cv2.waitKey(0)