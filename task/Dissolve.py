import cv2

# dissolve(source1,source2,starttime,dissolvetime)
def dissolve(vf, vt, st, cngtime):
    # 코덱 설정
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # video output 설정
    out = cv2.VideoWriter('output.avi', fourcc, 25.0, (640, 480))
    # 첫번째 인자는 저장할 비디오 파일 이름 두번째 인자는 코덱 설정 , 3번째 인자는 초당 frame ,4번째 인자는 해상도
    ret1, frame1 = vf.read()  # 1번 비디오 read
    ret2, frame2 = vt.read()  # 2번 비디오 read
    startTime = float(st)  # 시작시간
    fadeTime = float(cngtime)  # 전환시간
    w = 0.0
    while (vt.isOpened()):
        if (startTime > 0):
            startTime -= 25 / 1000  # 시작시간 타이머
        if (startTime <= 0 and fadeTime > 0):
            fadeTime -= 25 / 1000  # 전환시간 타이머
            w = float(cngtime - fadeTime) / float(cngtime)  # 전환시간에 따른 α값 설정
        ret1, frame1 = vf.read()  # 1번 비디오 read
        if (startTime < 0):
            ret2, frame2 = vt.read()  # 시작시간이 되면 2번 비디오 read
        if (w <= 1):
            res = cv2.addWeighted(frame1, float(1.0 - w), frame2, float(w), 0)  # α값 처리하여 비디오 합성

        else:
            res = frame2  # α값이 1보다 커지면 2번 비디오만 출력
            out.write(res)  # 비디오 write
        cv2.imshow('WindowName', res)  # Show
        key = cv2.waitKey(25)  # waitKey
        if (key == 27):  # esc일 시 break;
            break
    cv2.destroyAllWindows()
