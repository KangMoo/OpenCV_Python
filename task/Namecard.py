import numpy as np
import cv2
# imread
img_original = cv2.imread('namecard_05.jpg')
imgsize = 1000
# resize & initialize
img_resize = cv2.resize(img_original, (imgsize, int(img_original.shape[0] * (imgsize / img_original.shape[1]))),
                    interpolation=cv2.INTER_CUBIC)
img_show = np.copy(img_resize)
img_conv = np.copy(img_resize)
result = np.hstack((img_resize, img_conv))
cv2.namedWindow('CardName')
lowTH = 80
highTH = 150
lineTH = 0
pointLU = [0,0]
pointLB = [0,0]
pointRU = [0,0]
pointRB = [0,0]
nowPoint = pointLU
nowPointNum = 0
tryonce = False
def callbackConvrt(x):
    if x == 1:
        cv2.setTrackbarPos('Apply','CardName',0)
        applyImg()
def detectObject(img):
    global pointLU
    global pointLB
    global pointRU
    global pointRB
    img = cv2.bilateralFilter(img, 9, 25, 175)

    # morphology
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    img = cv2.erode(img, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    img = cv2.dilate(img, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    img = cv2.erode(img, kernel, iterations=1)

    # apply canny
    img = cv2.Canny(img, lowTH, highTH)
    # hough
    # detect 4 line
    lineTH = 0
    lines = cv2.HoughLines(img, 1, np.pi / 180, lineTH)
    xyset = np.zeros((4,4))
    count = 0
    while count < 4:
        lines = cv2.HoughLines(img, 1, np.pi / 180, lineTH)
        if lines is None:
            lineTH -= 3
            continue
        else:
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + imgsize * (-b) * 2)
                y1 = int(y0 + imgsize * (a) * 2)
                x2 = int(x0 - imgsize * (-b) * 2)
                y2 = int(y0 - imgsize * (a) * 2)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)
                xyset[count][0] = x1
                xyset[count][1] = y1
                xyset[count][2] = x2
                xyset[count][3] = y2
            count = count + 1
            lineTH = 0
    xyset = xyset.astype(int)
    for i in range(xyset.shape[0]):
        cv2.line(img_show, (xyset[i][0], xyset[i][1]), (xyset[i][2], xyset[i][3]), (0, 0, 255), 2)
    # 꼭지점 검출
    points = np.zeros((4,2))
    count = 0
    xyset = xyset.astype(float)
    for x in range(0,4):
        x1 = xyset[x][0]
        y1 = xyset[x][1]
        x2 = xyset[x][2]
        y2 = xyset[x][3]
        for y in range(x,4):
            x3 = xyset[y][0]
            y3 = xyset[y][1]
            x4 = xyset[y][2]
            y4 = xyset[y][3]
            if (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4) == 0: continue
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            if px > -100 and py > -100:
                if px < img.shape[1]+100 and py < img.shape[0]+100:
                    points[count][0] = px
                    points[count][1] = py
                    count = count + 1

    points = points.astype(int)
    # 좌 우 구분
    Lpoints = np.zeros((2,2))
    Rpoints = np.zeros((2,2))
    smallXpoint1 = [img.shape[1]+100,0]
    smallXpoint2 = [img.shape[1]+100,0]
    bigXpoint1 = [-100,0]
    bigXpoint2 = [-100,0]
    for x in range(0,points.shape[0]):
        if(points[x][0] < smallXpoint1[0]):
            smallXpoint1 = np.copy(points[x])
        if(points[x][0] > bigXpoint1[0]):
            bigXpoint1 = np.copy(points[x])

    for x in range(0,points.shape[0]):
        if(points[x][0] < smallXpoint2[0]):
            if np.all(points[x][0] != smallXpoint1):
                smallXpoint2 = np.copy(points[x])
        if(points[x][0] > bigXpoint2[0]):
            if np.all(points[x][0] != bigXpoint1):
                bigXpoint2 = np.copy(points[x])
    if smallXpoint1[1] > smallXpoint2[1]:
        pointLB = smallXpoint1
        pointLU = smallXpoint2
    else:
        pointLU = smallXpoint1
        pointLB = smallXpoint2
    if bigXpoint1[1] > bigXpoint2[1]:
        pointRB = bigXpoint1
        pointRU = bigXpoint2
    else:
        pointRU = bigXpoint1
        pointRB = bigXpoint2

    # 예외처리
    for x in range(0,2):
        if pointLB[x] < 0: pointLB[x] = 0
        if pointRB[x] < 0: pointRB[x] = 0
        if pointLU[x] < 0: pointLU[x] = 0
        if pointRU[x] < 0: pointRU[x] = 0

        if pointLB[x] > img_resize.shape[(x+1)%2]: pointLB[x] = img_resize.shape[(x+1)%2]
        if pointRB[x] > img_resize.shape[(x+1)%2]: pointRB[x] = img_resize.shape[(x+1)%2]
        if pointLU[x] > img_resize.shape[(x+1)%2]: pointLU[x] = img_resize.shape[(x+1)%2]
        if pointRU[x] > img_resize.shape[(x+1)%2]: pointRU[x] = img_resize.shape[(x+1)%2]


    #cv2.circle(img_show, (pointLU[0],pointLU[1]), 5, (255, 0, 255), -1)
    #cv2.circle(img_show, (pointLB[0],pointLB[1]), 5, (255, 0, 255), -1)
    #cv2.circle(img_show, (pointRU[0],pointRU[1]), 5, (255, 0, 255), -1)
    #cv2.circle(img_show, (pointRB[0],pointRB[1]), 5, (255, 0, 255), -1)
    return img

def transform2orthogonal(img):

    xsize = img_resize.shape[1]
    ysize = img_resize.shape[0]
    xwt = img_original.shape[1]/img_resize.shape[1]
    ywt = img_original.shape[1]/img_resize.shape[0]

    pts1 = np.float32((pointLU, pointLB, pointRU, pointRB))
    pts2 = np.float32([[0,0],[0,ysize],[xsize,0],[xsize,ysize]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    img = cv2.warpPerspective(img,M,(xsize,ysize))
    # 이미지 크기 처리
    img = cv2.resize(img, (pointRB[0]-pointLB[0], pointLB[1] - pointLU[1]),
                            interpolation=cv2.INTER_CUBIC)
    if img.shape[0] > img.shape[1]:
        cvtysize = int(img_original.shape[0] * (imgsize / img_original.shape[1]))
        img = cv2.resize(img, (int(img.shape[1] * (cvtysize / img.shape[0])), cvtysize), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, (imgsize, int(img.shape[0] * (imgsize / img.shape[1]))), interpolation=cv2.INTER_CUBIC)

    img_null = np.copy(img_resize)
    for x in range(0,img_null.shape[1]):
        if(x < img.shape[1]):
            for y in range(0, img_null.shape[0]):
                if(y < img.shape[0]):
                    img_null[y][x] = img[y][x]
                else:
                    img_null[y][x] = 0
        else:
            for y in range(0, img_null.shape[0]):
                img_null[y][x] = 0

    img = img_null
    return img

def filtering(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    return img
def drawLinesCircles(img):
    global tryonce
    if tryonce == False: return img
    cv2.line(img, (pointLU[0], pointLU[1]), (pointLB[0], pointLB[1]), (0, 0, 255), 2)
    cv2.line(img, (pointLU[0], pointLU[1]), (pointRU[0], pointRU[1]), (0, 0, 255), 2)
    cv2.line(img, (pointRB[0], pointRB[1]), (pointLB[0], pointLB[1]), (0, 0, 255), 2)
    cv2.line(img, (pointRB[0], pointRB[1]), (pointRU[0], pointRU[1]), (0, 0, 255), 2)
    cv2.circle(img, (pointLU[0], pointLU[1]), 10, (0, 255, 0), -1)
    cv2.circle(img, (pointRU[0], pointRU[1]), 10, (0, 255, 0), -1)
    cv2.circle(img, (pointLB[0], pointLB[1]), 10, (0, 255, 0), -1)
    cv2.circle(img, (pointRB[0], pointRB[1]), 10, (0, 255, 0), -1)
    cv2.circle(img, (nowPoint[0], nowPoint[1]), 10, (0, 255, 255), -1)

    return img
def applyImg():
    global img_resize
    global img_conv
    global img_show
    global lowTH
    global lineTH
    global highTH
    global tryonce
    img_conv = np.copy(img_resize)
    img_show = np.copy(img_resize)
    if tryonce == False:
        img_conv = detectObject(img_conv)
        tryonce = True


    img_conv = np.copy(img_resize)
    img_conv = transform2orthogonal(img_conv)

    img_conv = filtering(img_conv)

    # img_conv = cv2.cvtColor(img_conv, cv2.COLOR_GRAY2BGR)
    result = np.hstack((img_show, img_conv))

    result = cv2.resize(result, (imgsize, int(result.shape[0] * (imgsize / result.shape[1]))),
                        interpolation=cv2.INTER_CUBIC)
    cv2.imshow('CardName', result)
cv2.createTrackbar('Apply','CardName',0,1,callbackConvrt)
# cv2.createButton('button',callbackButton)

# add 2 Img
result = np.hstack((img_show, img_conv))
# resize
result = cv2.resize(result, (imgsize, int(result.shape[0] * (imgsize / result.shape[1]))),
                              interpolation=cv2.INTER_CUBIC)
cv2.imshow('CardName',result)
while(True):
    inputKey = cv2.waitKey(1)
    if(inputKey&0xFF == 27): break
    elif (inputKey == ord('w')):
        nowPoint[1] = nowPoint[1] - 1
    elif (inputKey == ord('a')):
        nowPoint[0] = nowPoint[0] - 1
    elif (inputKey == ord('s')):
        nowPoint[1] = nowPoint[1] + 1
    elif (inputKey == ord('d')):
        nowPoint[0] = nowPoint[0] + 1
    elif (inputKey == ord('f')):
        nowPointNum = (nowPointNum+1)%4
    if nowPointNum == 0: nowPoint = pointLU
    elif nowPointNum == 1: nowPoint = pointRU
    elif nowPointNum == 2: nowPoint = pointLB
    elif nowPointNum == 3: nowPoint = pointRB
    img_show = np.copy(img_resize)
    img_show = drawLinesCircles(img_show)
    # add 2 Img
    result = np.hstack((img_show, img_conv))
    # resize
    result = cv2.resize(result, (imgsize, int(result.shape[0] * (imgsize / result.shape[1]))),
                        interpolation=cv2.INTER_CUBIC)
    cv2.imshow('CardName', result)
cv2.destroyAllWindows()

