import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv2ImgTest():

    #이미지 읽기
    #cv2.imread(경로,색상)
    #색상종류3개 :
    #cv2.IMREAD_COLOR (컬러로 읽기) -> 1
    #cv2.IMREAD_GRAYSCALE (회색으로 읽기) -> 0
    #cv2.IMREAD_UNCHANGED (알파값까지 읽기) -> -1
    img = cv2.imread('test.jpg')

    # 이미지 반전,  0:상하, 1 : 좌우
    img = cv2.flip(img, 0)
    # cv2.waitKey() 는 keyboard입력을 대기하는 함수로 0이면 key입력까지 무한대기이며 특정 시간동안 대기하려면 milisecond값을 넣어주면 됩니다.

    #이미지 보기
    #cv2.imshow(윈도우명,이미지) : 이미지를 사이즈에 맞게 보여줌줌
    cv2.imshow('image',img)
    #키값 읽기
    key = cv2.waitKey(0)

    if key == 27: #27은 아스키코드로 ESC
        #화면에 나타난 윈도우 종료
        cv2.destroyAllWindows()
    elif key == ord('s'):
        #이미지 저장
        #cv2.imwrite(경로,이미지)
        cv2.imwrite('test.png',img)
        cv2.destroyAllWindows()

def matplotImgTest():
    img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
    img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
    img3 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.xticks([])  # x축 눈금
    plt.yticks([])  # y축 눈금
    plt.show()

def cv2VideoTest():
    windowName = "Live webcam"
    cv2.namedWindow(windowName)

    # cap 이 정상적으로 open이 되었는지 확인하기 위해서 cap.isOpen() 으로 확인가능
    cap = cv2.VideoCapture(0) # 0은 기본카메라
    # 파일로부터 읽기 : cap2 = cv2.VideoCapture(경로명)
    # cap.get(prodId)/cap.set(propId, value)을 통해서 속성 변경이 가능.

    filename = 'output.avi'
    codec = cv2.VideoWriter_fourcc('X','V','I','D')
    framerate = 30
    resolution = (640,480)
    VideoFileOutput = cv2.VideoWriter(filename,codec,framerate,resolution)
    # cv2.VideoWriter(output경로, 코덱, 프레임, 사이즈)

    print ('width: {0}, height: {1}'.format(cap.get(3), cap.get(4)))
    # 화면 크기 전환 3은 width, 4는 height
    # resolution과 다를 경우 저장이 안됨!!
    cap.set(3, 640)
    cap.set(4, 480)


    if cap.isOpened():
        # ret : frame capture결과(boolean)
        # frame : Capture한 frame
        ret, frame = cap.read()
    else:
        ret = False
    while ret:
        ret, frame = cap.read()
        # image를 BGR->RGB Convert함.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,0)
        # 저장
        VideoFileOutput.write(frame)
        # 저장시 값 변형(필터추가,크기변경,코덱변경 등)이 일어나면 저장이 안됨

        cv2.imshow(windowName,frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    VideoFileOutput.release()
    cap.release()


def trackbar():
    #callback 함수
    #Trackbar의 값이 변할때 실행되며, 값이 x로 들어옴
    def nothing(x):
        pass

    img = np.zeros((300, 512, 3), np.uint8)
    #윈도우 생성
    cv2.namedWindow('image')

    # trackbar를 생성하여 named window에 등록
    # cv2.createTrackbar(트랙바 이름, 윈도우 이름, 초기값, Max값(초기값은 항상 0),callback함수)
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)

    switch = '0:OFF\n1:On'
    cv2.createTrackbar(switch, 'image', 1, 1, nothing)

    while (1):
        cv2.imshow('image', img)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        # cv2.getTrackbarPos(트랙바 이름, 윈도우 이름)
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')

        if s == 0:
            img[:] = 0  # 모든 행/열 좌표 값을 0으로 변경. 검은색
        else:
            img[:] = [b, g, r]  # 모든 행/열 좌표값을 [b,g,r]로 변경

    cv2.destroyAllWindows()


######################################################################################################
#이하 실험용

