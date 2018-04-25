import cv2
import numpy as np

img_origin = cv2.imread('test.jpg',0)
#이미지 초기화
img1 = np.copy(img_origin)
img2 = np.copy(img_origin)
img3 = cv2.Canny(img_origin,0,0)
img_result = 0
maskSize = 0
lowTh = 0
highTh = 0
winname = 'Blur,Sharpening,Canny'
cv2.namedWindow(winname,cv2.WINDOW_AUTOSIZE)

# 마스크 사이즈 set callback함수
def setmasksize(x):
    global  maskSize
    global img1
    global img2
    global img_origin
    # 사이즈 저장
    maskSize = x
    if(maskSize == 0):
        img1 = img_origin
        img2 = img_origin
        # 결과 출력 함수
        makeResult()
        return

    # blur
    # box blur
    kernel = np.ones((maskSize * 2 + 1, maskSize * 2 + 1))
    # kernel 타입 변환
    kernel = kernel.astype('float64')
    # 총 합 1로 조정(밝기 유지)
    kernel[:] = kernel[:] / sum(sum(kernel))
    # img1에 필터 적용
    img1 = cv2.filter2D(img_origin,-1,kernel)
    # sharp
    kernel = np.ones((maskSize * 2 + 1, maskSize * 2 + 1))
    a = 0
    for x in range(0, maskSize * 2 + 1):
        b = 0
        for y in range(0, maskSize * 2 + 1):
            kernel[x, y] = a + b
            if y < maskSize:
                b = b + 1
            else:
                b = b - 1
        if x < maskSize:
            a = a + 1
        else:
            a = a - 1
    kernel = -kernel
    kernel[maskSize, maskSize] = 0
    kernel[maskSize, maskSize] = -sum(sum(kernel)) * 1.5
    # kernel 타입 변환
    kernel = kernel.astype('float64')
    # 총 합 1로 조정(밝기 유지)
    kernel[:] = kernel[:] / sum(sum(kernel))
    # img2에 필터적용
    img2 = cv2.filter2D(img_origin,-1,kernel)
    # 결과 출력 함수
    makeResult()
def setlowTH(x):
    global lowTh
    global img3
    # lowThreshold저장
    lowTh = x
    # 캐니 함수 실행
    img3 = cv2.Canny(img_origin, lowTh, highTh)
    # 결과 출력 함수
    makeResult()
def sethighTh(x):
    global highTh
    global img3
    # hightThreshold저장
    highTh = x
    # 캐니 함수 실행
    img3 = cv2.Canny(img_origin, lowTh, highTh)
    # 결과 출력 함수
    makeResult()
def makeResult():
    global img1
    global img2
    global img3
    global img_result
    # 결과물 한 이미지로 합치기
    temp = np.hstack((img_origin, img1))
    temp2 = np.hstack((img2, img3))
    img_result = np.vstack((temp, temp2))
    cv2.imshow(winname, img_result)

#트랙바 설정
cv2.createTrackbar('MaskSize',winname,0,10,setmasksize)
cv2.createTrackbar('LowTH',winname,0,200,setlowTH)
cv2.createTrackbar('HighTH',winname,0,200,sethighTh)

# 이미지 합치기
temp = np.hstack((img_origin,img1))
temp2 = np.hstack((img2,img3))
img_result = np.vstack((temp,temp2))
cv2.imshow(winname,img_result)

# esc 입력시 종료
while 1:
    if cv2.waitKey() & 0xFF == 27:
        break
cv2.destroyAllWindows()