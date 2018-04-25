import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('test.jpg')
# 이미지 복제
img_masked = np.copy(img)
img_canny = np.copy(cv2.cvtColor(img_masked,cv2.COLOR_BGR2GRAY))
lowset = np.copy(img_canny)
highset = np.copy(img_canny)

# Window name
windowName = 'Mask'
# Trackbar name 설정
MaskName = 'MaskType'
# Window name 설정
cv2.namedWindow(windowName,cv2.WINDOW_AUTOSIZE)
# 변수 초기화
maskType = 0
maskSize = 0
low_threshold = 0
high_threshold = 0
kernel = np.ones((maskSize*2+1,maskSize*2+1),'float64')
kernel = kernel.astype('float64')
#콜백함수
def mskcallback(x):
    # kernel 타입 저장
    global maskType
    maskType= x
    # 이미지 변환 함수 실행
    modifyImg()

#콜백함수
def sizecallback(x):
    #kernel사이즈 저장
    global maskSize
    maskSize= x
    # 이미지 변환 함수 실행
    modifyImg()
#콜백함수
def lowthcallback(x):
    global low_threshold
    #low_threadhold값 저장
    low_threshold = x
    # 이미지 변환 함수 실행
    modifyImg()

#콜백함수
def highthcallback(x):
    global high_threshold
    #high_threadhold값 저장
    high_threshold = x
    # 이미지 변환 함수 실행
    modifyImg()

#이미지 마스크처리 함수
def modifyImg():
    global img_masked
    global img_canny
    global img
    global maskType
    global low_threshold
    global high_threshold
    global highset
    global lowset
    #원본 이미지 복사
    img_masked = np.copy(img)
    #형변환
    img_masked = img_masked.astype('float64')
    #캐니 알고리즘은 따로 처리
    if maskType == 3:
        thresholdset()
    else:
        #kernel생성 함수 실행
        remakeMask()
        #마스크처리
        img_masked = cv2.filter2D(img,-1,kernel)
    # 이미지 예외처리 ~
    np.clip(img_masked, 0, 255, img_masked)
    cv2.imshow(windowName,img_masked)

#mask 설정 함수
def remakeMask():
    #kernel 설정
    global kernel
    global maskType
    global maskSize
    #original
    if(maskType == 0):
        kernel = np.zeros((maskSize * 2 + 1, maskSize * 2 + 1))
        kernel[maskSize][maskSize] = 1
    # blur
    elif(maskType == 1):
        # box blur
        kernel = np.ones((maskSize * 2 + 1, maskSize * 2 + 1))
    #sharp
    elif(maskType == 2):
        kernel = np.ones((maskSize * 2 + 1, maskSize * 2 + 1))
        if(maskSize == 0): return
        a = 0
        for x in range(0,maskSize*2+1):
            b = 0
            for y in range(0,maskSize*2+1):
                    kernel[x,y] = a+b
                    if y<maskSize:
                        b = b+1
                    else:
                        b = b-1
            if x < maskSize:
                a = a+1
            else:
                a = a-1
        kernel = -kernel
        kernel[maskSize,maskSize] = 0
        kernel[maskSize, maskSize] = -sum(sum(kernel))*1.5
    # kernel 타입 변환
    kernel = kernel.astype('float64')
    # 총 합 1로 조정(밝기 유지)
    kernel[:] = kernel[:]/sum(sum(kernel))

# canny 생성함수
def canny():
    global img_canny
    global img_masked
    global img
    global low_threshold
    global high_threshold
    global maskSize
    img_canny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 캐니 알고리즘 순서 : 노이즈 제거 -> 엣지검출 -> non-maximum 값 제거 -> double threadhold로 크기 구분 -> 엣지연결
    # 노이즈 제거~
    # 가우시안 필터 적용
    img_canny = cv2.GaussianBlur(img_canny,(maskSize*2+1,maskSize*2+1),0)
    cv2.imshow('step1',img_canny)
    # ~ 노이즈 제거

    # 엣지 검출 ~
    # 수직,수평 방향 검출 sobel mask 생성
    xsobel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    ysobel = np.array([[-1,-2,-1],
                       [0,0,0],
                       [1,2,1]])
    img_xsobel = cv2.filter2D(img_canny,-1,xsobel)
    img_ysobel = cv2.filter2D(img_canny,-1,ysobel)
    # maginutude
    img_canny = np.abs(img_xsobel)+np.abs(img_ysobel)
    cv2.imshow('step2', img_canny)
    # ~ 엣지검출

    # non-maximum ~
    temp = np.zeros(img_canny.shape)
    for x in range(0,img_canny.shape[1]-1):
        for y in range(0,img_canny.shape[0]-1):
            for i in range(0,4):
                flag = False
                # 0,45,90,135 각도에 따른 non-maximum검출
                if(i == 0):
                    temp2 = np.array([img_canny[y][x-1],img_canny[y][x],img_canny[y][x+1]])
                    if img_canny[y][x] >= np.amax(temp2):
                        temp[y][x] = img_canny[y][x]
                        flag = True
                        break
                elif(i==1):
                    temp2 = np.array([img_canny[y-1][x - 1], img_canny[y][x], img_canny[y+1][x + 1]])
                    if img_canny[y][x] >= np.amax(temp2):
                        temp[y][x] = img_canny[y][x]
                        flag = True
                        break
                elif(i==2):
                    temp2 = np.array([img_canny[y+1][x - 1], img_canny[y][x], img_canny[y-1][x + 1]])
                    if img_canny[y][x] >= np.amax(temp2):
                        temp[y][x] = img_canny[y][x]
                        flag = True
                        break
                elif(i==3):
                    temp2 = np.array([img_canny[y-1][x], img_canny[y][x], img_canny[y+1][x]])
                    if img_canny[y][x] >= np.amax(temp2):
                        temp[y][x] = img_canny[y][x]
                        flag = True
                        break;
            # 최고가 아닐 경우 0값 대입
            if flag == False:
                temp[y][x] = 0
    img_canny = temp.astype('uint8')
    cv2.imshow('step3', img_canny)
    # ~ non-maximum
    thresholdset()

def thresholdset():
    global img_canny
    global img_masked
    global img
    global loset
    global highset
    global low_threashold
    global high_threshold
    # double threadhold ~
    temp = np.copy(img_canny).flatten()
    temp2 = np.ones(img_canny.shape).flatten()*255
    lowset = np.where(temp>low_threshold,temp,0)
    highset = np.where(temp>high_threshold,temp2,0)
    lowset = lowset.reshape(img_canny.shape)
    highset = highset.reshape(img_canny.shape)

    cv2.imshow('lowset', lowset)
    cv2.imshow('highset', highset)
    # ~ double threadhold

    # 엣지 연결 ~
    for x in range(0, img_canny.shape[1] - 1):
        for y in range(0, img_canny.shape[0] - 1):
            if lowset[y][x] != 0:
                temp = np.array([[highset[y - 1][x - 1], highset[y - 1][x], highset[y - 1][x + 1]],
                                  [highset[y][x - 1], highset[y][x], highset[y][x + 1]],
                                  [highset[y + 1][x - 1], highset[y + 1][x], highset[y + 1][x + 1]]])
                if np.amax(temp)>0:
                    highset[y][x] = 255
    img_masked = np.copy(highset)
    # ~ 엣지 연결

# set trackbar
cv2.createTrackbar(MaskName,'Mask',0,3,mskcallback)
cv2.createTrackbar('Size','Mask',0,10,sizecallback)
cv2.createTrackbar('Low_th','Mask',0,100,lowthcallback)
cv2.createTrackbar('High_th','Mask',0,100,highthcallback)
canny()
# imshow
cv2.imshow('Mask',img)
#캐니 함수 실행하여 highset,lowset저장


# esc입력 시 break
while(1):
    if cv2.waitKey(0) & 0xFF == 27:
        break
# 종료
cv2.destroyAllWindows()