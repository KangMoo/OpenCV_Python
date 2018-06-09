import cv2
import numpy as np

# 변수 선언 및 초기화
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
hand_cascade = cv2.CascadeClassifier('output.xml')
faces = ()
hands = ()
faceRemainTimer = 0
handRemainTimer = 0

# 얼굴영역 검출 함수
def faceDetect(img):
    global faces
    global hands
    global faceRemainTimer
    global handRemainTimer
    global frameTimer
    # 전처리 ~
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    gray = cdf[gray]
    # ~ 전처리
    # 영역검출
    faces = face_cascade.detectMultiScale(gray,1.3,3)
    hands = hand_cascade.detectMultiScale(gray,1.3,3)
    return img

# 얼굴 확대 함수
def zoomFace(img):
    global hands
    global faces


    handCenter = (int(hands[0][0]+(hands[0][2]/2)),int(hands[0][1]+(hands[0][3]/2)))
    closestFace = faces[0]
    cfaceCenter = (int(closestFace[0] + (closestFace[2] / 2)) , int(closestFace[1] + (closestFace[3] / 2)) )

    # 검출된 손이랑 가장 가까운 얼굴 탐색
    for i in range(0,faces.shape[0]):
        ifaceCenter = (int(faces[i][0]+(faces[i][2]/2)),int(faces[i][1]+(faces[i][3]/2)))
        if (handCenter[0]-ifaceCenter[0])**2 + (handCenter[1]-ifaceCenter[1])**2 < (handCenter[0]-cfaceCenter[0])**2 + (handCenter[1]-cfaceCenter[1])**2:
            closestFace = faces[i]
            cfaceCenter = (int(closestFace[0] + (closestFace[2] / 2)) , int(closestFace[1] + (closestFace[3] / 2)))

    # 손이랑 가장 가까운 얼굴 영역 return
    img2 = img[closestFace[1]:(closestFace[1]+closestFace[3]),closestFace[0]:(closestFace[0]+closestFace[2])]
    return img2


winName = "Face Detection"
cv2.namedWindow(winName)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
resolution = (frame.shape[1],frame.shape[0])

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False
while ret:
    ret, frame = cap.read()
    # 빠른 탐색을 위한 이미지 크기 축소
    resizerate = 2
    frame2 = cv2.resize(np.copy(frame), (int(frame.shape[1]/resizerate),int(frame.shape[0]/resizerate)), interpolation=cv2.INTER_CUBIC)
    faceDetect(frame2)

    # 축소시킨 크기 만큼 다시 키우기
    faces = faces[:]*resizerate
    hands = hands[:]*resizerate
    frame3 = ()
    # 손&얼굴이 검출 되었다면 얼굴 확대
    if np.array(hands).size > 0 and np.array(faces).size > 0:
        frame3 = zoomFace(np.copy(frame))
        frame3 = cv2.resize(frame3,(frame3.shape[1]*2,frame3.shape[0]*2),interpolation = cv2.INTER_CUBIC)
        cv2.imshow('CroppedImg',frame3)
    # 검출된 얼굴영역 모두 사각형 처리
    for (x, y, w, h) in faces:
         cv2.rectangle(frame, (x , y ),
                     (x  + w , y  + h ), (255, 0, 0), 2)
    # 검출된 손바닥영역 모두 사각형 처리
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x , y ),
                    (x  + w , y  + h ), (0, 255, 0), 2)
    cv2.imshow(winName, frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()