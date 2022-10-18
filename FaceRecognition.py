import cv2

facecascade = cv2.CascadeClassifier('project\detect\haarcascade_frontalface_alt.xml')

def draw_border(img,color,feature,clf): # set frame size
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    scale = []
    text = ['Name','Unknown']
    for (x,y,w,h) in feature:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,5)
        id,con = clf.predict(gray[y:y+h,x:x+w])
        con = (round(100-con))
        if id == 2:
            if con >=25:
                cv2.putText(img,text[0]+' '+str(con)+' %',(int(x+w/2)-len(text[0])*12,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                print('Face Detected')
            else:
                cv2.putText(img,text[1]+' '+str(con)+' %',(int(x+w/2)-len(text[0])*12,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    return img

def detect(img,facecascade,clf): # Face Detection
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faceborder = facecascade.detectMultiScale(gray,1.1,17)
    img= draw_border(img,(255,0,0),faceborder,clf)
    return img

cap = cv2.VideoCapture(0)
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read('myface.xml')

while(True):
    ret,frame = cap.read()
    frame = detect(frame,facecascade,clf)
    cv2.imshow('facedetection',frame)

    if(cv2.waitKey(1) & 0xFF == ord('e')):
        break

cap.release()
cv2.destroyAllWindows()
 