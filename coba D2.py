import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = PoseDetector()
while True :
    # membaca video dari webcam
    success, img = cap.read()
    img = detector.findPose(img)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q') : break

cap.release()
cv2.destroyAllWindows()