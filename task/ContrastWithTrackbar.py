import cv2
import numpy as np

def tracbar(orgimg):
    #명도,switch 변수 선언
    cont = 0
    switch = 0
    def nothing(x):
        pass

    def callback_contrast(x):
        #명도 저장
        nonlocal cont
        cont = x - 100
        #값 변화에 따른 이미지 변환
        img_remake()

    def callback_switch(x):
        #switch값 저장
        nonlocal switch
        switch = x
        #값 변화에 따른 이미지 변환
        img_remake()

    def img_remake():
        nonlocal switch
        nonlocal cont
        nonlocal img_ed
        nonlocal img
        #원본 이미지 복사
        img_ed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_ed = img_ed.astype('float64')
        # 명도조절 ~
        #선형
        if switch == 0:
            if cont > 0:
                for i in range(0, cont):
                    img_ed[:] = img_ed[:] * float(1 + 1 / 128)
            elif cont < 0:
                for i in range(0, - cont):
                    img_ed[:] = img_ed[:] * float(1 - 1 / 128)
        #S형
        else:
            if cont > 0:
                img_ed[:] = img_ed[:] * float(1 + cont / 128)
            elif cont < 0:
                img_ed[:] = img_ed[:] * float(1 + cont / 128)
        # ~ 명도조절
        #이미지 예외처리 ~
        np.clip(img_ed,0,200,img_ed)
        # ~ 이미지 예외처리
        img_ed = img_ed.astype('uint8')
        #이미지 출력
        cv2.imshow('image', img_ed)
    #이미지 읽기
    img = orgimg
    #이미지 복사
    img_ed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #배열의 형변환
    img_ed = img_ed.astype('float64')
    #윈도우 이름 설정
    cv2.namedWindow('image')
    cv2.createTrackbar('Contrast', 'image', 100, 200, callback_contrast)
    switch = 'LINE or S'
    cv2.createTrackbar(switch, 'image', 1, 1, callback_switch)
    while(True):
        #이미지 출력
        cv2.imshow('image', img_ed.astype('uint8'))
        #esc입력 시 종료
        if cv2.waitKey(0) & 0xFF == 27: break
    #윈도우 종료
    cv2.destroyAllWindows()