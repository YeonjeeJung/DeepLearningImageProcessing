import cv2
import numpy as np

def CaptureVideo(input_filename):

    # 비디오 캡쳐 시작
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 16
    size = (int(cap.get(3)), int(cap.get(4)))
    #size = (100,100)

    out = cv2.VideoWriter(input_filename, fourcc, fps, size)
    print("Press ESC to stop Recording...")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            out.write(frame)

            cv2.imshow(input_filename, frame)

            k = cv2.waitKey(1)
            if k == 27:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

"""
def ReadVideo(input_filename):
    names = ["None",]
    frames = [[],]

    print("Start Reading Video")
    print("Press N to start labeling...")

    # 비디오 읽기
    vcap = cv2.VideoCapture(input_filename)
    cmd = False
    thisname = None
    i = 1

    center_w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    center_h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
    crop_size = 480 // 2

    while vcap.isOpened():
        ret, frame = vcap.read()

        if ret:
            #print(cmd)
            cv2.imshow('frame',frame)
            crop_frame = frame[center_h-crop_size:center_h+crop_size, center_w-crop_size:center_w+crop_size, :]

            if cmd:
                frames[i].append(crop_frame)
            if not cmd:
                frames[0].append(crop_frame)
        else:
            break

        if cv2.waitKey(1) == ord('n') and (not cmd):
            print("Pressed N, start Labeling")
            thisname = input("Write the object name : ")
            print("Press Q to stop labeling...")
            cmd = True
            names.append(thisname)
            frames.append([])

        elif cv2.waitKey(1) == ord('q') and cmd:
            print("Pressed Q, stop labeling")
            _ = input("Press Enter to Resume :")
            print("Press N to start labeling...")
            cmd = False
            i += 1

    vcap.release()
    cv2.destroyAllWindows()

    for k in range(len(frames)):
        frames[k] = np.array(frames[k])

    return (frames, names)
"""

def predictVideo(model, labels):
    vcap = cv2.VideoCapture(0)
    thispredict = None
    temppredict = None

    center_w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    center_h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
    crop_size = 480 // 2

    while vcap.isOpened():
        ret, frame = vcap.read()

        if ret:
            cv2.imshow('frame',frame)
            crop_frame = frame[center_h-crop_size:center_h+crop_size, center_w-crop_size:center_w+crop_size, :]

            crop_frame = crop_frame.reshape(1, *crop_frame.shape)

            temp = model.predict(crop_frame)
            temp = temp[0]

            #print(temp)
            #ind = np.where(temp==1.0)
            #print(temp)
            ind = np.where(temp==max(temp))
            #print(ind)
            #if len(ind[0]) != 0:
                #print(temp)
                #print(np.where(temp==1.0))
            temppredict = labels[ind[0][0]]
            #temppredict = labels[ind[0]]
                #print(temppredict)

            if thispredict != temppredict:
                #if temppredict != 'None':
                print(temppredict)
                thispredict = temppredict

            k = cv2.waitKey(1)
            if k == 27:
                break
        else:
            break

    vcap.release()
    cv2.destroyAllWindows()

col, width, row, height = -1, -1, -1, -1
frame = None
frame2 = None
inputmode = False
rectangle = False
trackWindow = None
roi_hist = None
#window_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
#window_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def onMouse(event, x, y, flags, param):
    global col, width, row, height, frame, frame2, inputmode, window_width, window_height
    global rectangle, roi_hist, trackWindow

    if inputmode:
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            col, row = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle:
                frame = frame2.copy()
                cv2.rectangle(frame, (col, row), (x, y), (0, 255, 0), 2)
                cv2.imshow('frame', frame)
        elif event == cv2.EVENT_LBUTTONUP:
            inputmode = False
            rectangle = False
            #cv2.rectangle(frame, (col, row), (x, y), (0, 255, 0), 2)
            cv2.rectangle(frame, (col, row), (480, 480), (0, 255, 0), 2)
            #height, width = abs(row-y), abs(col-x)
            height, width = 480, 480
            trackWindow = (col, row, width, height)
            roi = frame[row:row+height, col:col+width]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    return

def camShift(input_filename):
    global frame, frame2, inputmode, trackWindow, roi_hist
    labels = []
    frames = []

    cap = cv2.VideoCapture(input_filename)

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouse, param=(frame, frame2))

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.1)

    i = -1
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if trackWindow is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, trackWindow = cv2.meanShift(dst, trackWindow, termination)
            x, y, w, h = trackWindow
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            thisframe = frame[y:y+w, x:x+h]
            frames[i].append(thisframe)
            #print(thisframe.shape)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('i'):
            label = input("Write the label : ")
            labels.append(label)
            frames.append([])
            i += 1
            print('Select Area for CamShift and Enter a key')
            inputmode = True
            frame2 = frame.copy()

            while(inputmode):
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

        elif k == ord('q'):
            inputmode = False
            trackWindow = None


    cap.release()
    cv2.destroyAllWindows()

    for i in range(len(frames)):
        frames[i] = np.array(frames[i])
    return (frames, labels)
