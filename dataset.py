import cv2
faceCascade =  cv2.CascadeClassifier("project\detect\haarcascade_frontalface_alt.xml")

def createDataSet(img,id,img_id):
   cv2.imwrite('project\Face\datapic\pic.'+str(id)+'.'+str(img_id)+'.jpg',img)

def DrawingFace(img,color,text,features):
    scale = []
    for(x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(int(x+w/2)-len(text)*8,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        scale = [x,y,w,h]
    return img,scale

def detection(img,faceCascade,pic_no):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faceboarder=faceCascade.detectMultiScale(gray,1.1,10)
    img,scale=DrawingFace(img,(255,0,0),"Face",faceboarder)
    if len(scale) == 4:
        id = 2
        result = img[scale[1]:scale[1]+scale[3],scale[0]:scale[0]+scale[2]]
        createDataSet(result,id,pic_no)
    return img

cap = cv2.VideoCapture(0)
pic_no = 1

while(True):
    if pic_no<=200:
        ret,frame=cap.read()
        frame = detection(frame,faceCascade,pic_no)
        pic_no += 1
    else:
        break
    cv2.imshow('FaceRecogDetect',frame)
    if(cv2.waitKey(1) & 0xFF == ord('e')):
        break

cap.release()
cv2.destroyAllWindows