from scipy.spatial import distance as dist
import face_recognition
from threading import Thread
import cv2
import numpy as np
import playsound
 
min_ear=0.28
Eye_Frames=10
counter=0
ALARM_ON=False
def alarm(soundfile):
    playsound.playsound(soundfile)
    playsound.PlaysoundException()
def eye_aspect_ratio(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])
    EyeAspectRatio=(A+B)/(2*C)
    return EyeAspectRatio

def main():
    global counter,ALARM_ON
    video_capture=cv2.VideoCapture(0)
    video_capture.set(3,640)
    video_capture.set(4,480)
    while True:
        ret,frame=video_capture.read()
        face_landmarks_list=face_recognition.face_landmarks(frame)
        for face_landmark in face_landmarks_list:

            leftEye=face_landmark['left_eye']
            rightEye=face_landmark['right_eye'] 
            left_EAR=eye_aspect_ratio(leftEye)
            right_EAR=eye_aspect_ratio(rightEye)

            ear=(left_EAR+right_EAR)/2
            
            lpts=np.array(leftEye)
            rpts=np.array(rightEye)

            cv2.polylines(frame,[lpts],True,(255,255,0),1)
            cv2.polylines(frame,[rpts],True,(255,255,0),1)
            if ear<min_ear:
                counter +=1
                if counter>=Eye_Frames:
                    if not ALARM_ON:
                        ALARM_ON=True
                        t=Thread(target=alarm,args=('Alarm.mp3',))
                        t.daemon=True
                        t.start()    
                cv2.putText(frame,"!!!!ALERT!!!!,you are feeling asleep",(5,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
                
            else:
                counter=0
                ALARM_ON=False
        cv2.putText(frame,"EAR: {:.2f}".format(ear),(300,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        cv2.imshow("@@@ Drowsiness Detector Window @@@",frame)
        if cv2.waitKey(1)==ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()
