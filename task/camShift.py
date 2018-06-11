import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
rows, cols = frame.shape[:2]
startTracking = False
inputmode = False
frameCopy = None
stopFrame = False
cropImg = None
MIN_MATCH_COUNT = 10
MAX_MATCH_THRESH = 35
# 프레임 중심에서 초기 창 위치 정의
windowWidth = 1
windowHeight = 1
windowCol = -10
windowRow = -10
orb = cv2.ORB_create()
window = (windowCol, windowRow, windowWidth, windowHeight)
kp1 =None
des1 = None
kp2 = None
des2 = None
img3 = None

# 종료 기준 설정 : 완료 10 반복 또는 1 픽셀 미만으로 이동
terminationCriteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS , 10, 1)
def onMouse(event,x,y,flags,param):
    global windowCol, windowRow,windowWidth,windowHeight, inputmode,cropImg,kp1,des1
    global window,roi,lowLimit,highLimit,mask,roiHist,terminationCriteria,frameCopy,frame,startTracking,stopFrame
    if event == cv2.EVENT_LBUTTONDOWN:
        if inputmode == False:
            frameCopy = np.copy(frame)
            inputmode = True
            startTracking = False
            windowRow, windowCol = y, x
            windowWidth = 1
            windowHeight = 1
    if inputmode:
        if event == cv2.EVENT_LBUTTONDOWN:
            windowRow, windowCol = y, x
        if event == cv2.EVENT_MOUSEMOVE:
            windowWidth = x - windowCol
            windowHeight = y - windowRow
        elif event == cv2.EVENT_LBUTTONUP:
            window = (windowCol, windowRow, windowWidth, windowHeight)
            # roi 설정
            roi = frame[windowRow:windowRow + windowHeight, windowCol:windowCol + windowWidth]
            # HSV스케일 변환
            roiHsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # limit설정
            lowLimit = np.array((0., 60., 32.))
            highLimit = np.array((180., 255., 255.))
            mask = cv2.inRange(roiHsv, lowLimit, highLimit)

            # 마스킹되지 않은 영역의 색조 그래프 계산
            roiHist = cv2.calcHist([roiHsv], [0], mask, [180], [0, 180])
            cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            cropImg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[windowRow:windowRow + windowHeight, windowCol:windowCol + windowWidth]
            #cropImg = cv2.resize(cropImg,(500,int(cropImg.shape[0]*(500/cropImg.shape[1]))),interpolation=cv2.INTER_CUBIC)
            kp1, des1 = orb.detectAndCompute(cropImg, None)
            startTracking = True
            inputmode = False
            stopFrame = False


while True:
    if inputmode or stopFrame:
        frame = np.copy(frameCopy)
    else:
        ret , frame = cap.read()
    if ret:
        k = cv2.waitKey(60) & 0xff
        if startTracking:
            # 히스토그램 역 투영 계산
            frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backprojectedFrame = cv2.calcBackProject([frameHsv], [0], roiHist, [0, 180], 1)
            cv2.imshow('backprojectedFrame',backprojectedFrame)
            # 결과 향상을 위한 마스크
            mask = cv2.inRange(frameHsv, lowLimit, highLimit)
            backprojectedFrame &= mask
            # 새 창 위치를 가져 오는 임시 방법을 적용
            ret, window = cv2.CamShift(backprojectedFrame, window, terminationCriteria)

            # 프레임에 창 그리기
            points = cv2.boxPoints(ret)
            points = np.int0(points)

            #############################
            kp2, des2 = orb.detectAndCompute(frame, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            good = []
            for m in matches:
                if m.distance < MAX_MATCH_THRESH:
                    good.append(m)
            if (len(good) > MIN_MATCH_COUNT):

                for m in good:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()
                    h, w = cropImg.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = pts
                    if (M is not None):
                        dst = cv2.perspectiveTransform(pts, M)
                    #frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                    points = np.copy(dst)
                    print(dst.shape)
                    print(dst)

                    left = 1000
                    right = 0
                    up = 1000
                    down = 0
                    for x in range(np.array(dst).shape[0]):
                        if left > dst[x][0][0]:
                            left = dst[x][0][0]
                        if right < dst[x][0][0]:
                            right = dst[x][0][0]
                        if up > dst[x][0][1]:
                            up = dst[x][0][1]
                        if down < dst[x][0][1]:
                            down = dst[x][0][1]
                    #windowRow = dst[0][0][1]
                    #windowCol = dst[0][0][0]
                    #windowWidth = abs(dst[3][0][0] - dst[0][0][0])
                    #windowHeight = abs(dst[1][0][1] - dst[0][0][1])
                    #window = (windowCol, windowRow, windowWidth, windowHeight)
                    window = (left, up, right - left, down - up)
                    print(window)
            points = np.int0(points)
            frame = cv2.polylines(frame, [points], True, 255, 2)

            img3 = cv2.drawMatches(cropImg, kp1, frame, kp2, good, None, flags=2)
        else:
            points = np.array([[windowCol,windowRow],[windowCol+windowWidth,windowRow],[windowCol+windowWidth,windowRow+windowHeight],[windowCol,windowRow+windowHeight]])
            '''
            cv2.circle(frame, (windowCol, windowRow), 5, (255, 0, 0), 3)
            cv2.circle(frame, (windowCol+windowWidth,windowRow), 5, (0, 255, 0), 3)
            cv2.circle(frame, (windowCol+windowWidth,windowRow+windowHeight), 5, (0, 0, 255), 3)
            cv2.circle(frame, (windowCol,windowRow+windowHeight), 5, (255, 0, 255), 3)
            '''
            frame = cv2.polylines(frame, [points], True, 255, 2)
        if k== ord('s'):
            frameCopy = np.copy(frame)
            stopFrame = True
        cv2.setMouseCallback('camshift',onMouse)

        # 결과 프레임 표시
        cv2.imshow('camshift', frame)
        if img3 is not None:
            cv2.imshow('test',img3)
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
