import dlib
import cv2
import numpy
from imutils import face_utils
from scipy.spatial import distance as dist
import dotenv
import os


class DrowsinessDetector:
    def __init__(self):
        dotenv.load_dotenv()
        self._consecutiveDrowsyFrames = 0
        self._maxDrowsyFramesBeforeSignal = int(os.getenv("FRAMES_BEFORE_DROWSINESS_CONFIRMED"))
        self._minimumEyeAspectRatioBeforeCloseAssumed = float(os.getenv("MINIMUM_EYE_ASPECT_RATIO_BEFORE_ASSUMED_CLOSED"))


    def areEyesClosed(self, img):
	face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
	leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')	
	reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



	lbl=['Kapali','Acik']

	model = load_model('models/model.h5')
	path = os.getcwd()
	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_COMPLEX_SMALL
	count=0
	score=0
	thicc=2
	rpred=[99]
	lpred=[99]

	while(True):
		ret, frame = cap.read()
		height,width = frame.shape[:2] 

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
		faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
		left_eye = leye.detectMultiScale(gray)
		right_eye =  reye.detectMultiScale(gray)

		cv2.rectangle(frame, (0,height-50) , (200,height) , (0, 255, 0) , thickness=cv2.FILLED )

		for (x,y,w,h) in faces:
			cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0, 255, 0) , 1 )

		for (x,y,w,h) in right_eye:
			r_eye=frame[y:y+h,x:x+w]
			count=count+1
			r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
			r_eye = cv2.resize(r_eye,(24,24))
			r_eye= r_eye/255
			r_eye=  r_eye.reshape(24,24,-1)
			r_eye = np.expand_dims(r_eye,axis=0)
			rpred = model.predict_classes(r_eye)
			if(rpred[0]==1):
				return False 
			if(rpred[0]==0):
				return True
			break

    
        facialLandmarks = shapePredictor(img, dets[0])
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        shape = face_utils.shape_to_np(facialLandmarks)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.getEyeAspectRatio(leftEye)
        rightEAR = self.getEyeAspectRatio(rightEye)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
        ear = (leftEAR + rightEAR) / 2
        print(ear)

        if ear < self._getMinimumEyeAspectRatio():
            self.incrementNumberConsecutiveDrowsyFrames()
            return True

        return False

    def getEyeAspectRatio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    def isDrowsy(self):
        return self.getNumberConsecutiveDrowsyFrames() > \
               self.getMaxDrowsyFramesBeforeSignal()

    def _getMinimumEyeAspectRatio(self):
        return self._minimumEyeAspectRatioBeforeCloseAssumed

    def getMaxDrowsyFramesBeforeSignal(self):
        return self._maxDrowsyFramesBeforeSignal

    def getNumberConsecutiveDrowsyFrames(self):
        return self._consecutiveDrowsyFrames

    def incrementNumberConsecutiveDrowsyFrames(self):
        self._consecutiveDrowsyFrames += 1

    def resetNumberConsecutiveDrowsyFrames(self):
        self._consecutiveDrowsyFrames = 0

if __name__ == "__main__":
    d = DrowsinessDetector()
    img = dlib.load_grayscale_image("testImage.png")
    print(d.areEyesClosed(img))
