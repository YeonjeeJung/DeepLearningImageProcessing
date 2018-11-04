import cv2
import Video
import Learning
import numpy as np


"""
def predictVideo(model, labels):

    global frame, frame2, inputmode, trackWindow, roi_hist

    pcap = cv2.VideoCapture(0)

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouse, param=(frame, frame2))

    #termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    trackWindow = None

    while pcap.isOpened():
        ret, frame = pcap.read()

        if not ret:
            break


        if trackWindow is not None:

            x, y, w, h = trackWindow
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            thisframe = frame[y:y+w, x:x+h]
            thisframe = thisframe.reshape(1, *thisframe.shape)
            thispredict = model.predict(thisframe)
            print(thispredict)
            thispredict = thispredict[0]

            ind = np.where(thispredict==max(thispredict))
            #print(ind)
            #print(labels)
            temppredict = labels[ind[0][0]]
            print(temppredict)
            #print(thisframe.shape)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('i'):
            #label = input("Write the label : ")
            #labels.append(label)
            #frames.append([])
            #i += 1
            print('Select Area for CamShift and Enter a key')
            inputmode = True
            frame2 = frame.copy()

            while(inputmode):
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

        elif k == ord('q'):
            inputmode = False
            trackWindow = None

    pcap.release()
    cv2.destroyAllWindows()
"""

if __name__ == '__main__':

    Video.CaptureVideo('output.avi')

    _ = input("Press Enter to Read Video :")
    (frames, labels) = Video.camShift('output.avi')

    print(str(len(labels))+" objects detected : ",labels)
    #for i in range(len(frames)):
        #print(frames[i].shape)
    model = Learning.train(frames, labels, epoches=5, lr=0.0001)
    print(model)
    #predictVideo(model, labels)
    Video.predictVideo(model, labels)
