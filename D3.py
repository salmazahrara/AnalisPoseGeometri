import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi pose detector
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        enableSegmentation=False,
                        detectionCon=0.5,
                        trackCon=0.5)

while True:
    # Tangkap setiap frame dari webcam
    success, img = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera")
        break

    # Temukan pose manusia dalam frame
    img = detector.findPose(img)

    # Temukan landmark, bounding box, dan pusat tubuh
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    # Jika terdeteksi pose
    if lmList and bboxInfo:
        center = bboxInfo["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Hitung jarak antara landmark 11 dan 15
        length, img, info = detector.findDistance(lmList[11][0:2],
                                                  lmList[15][0:2],
                                                  img=img,
                                                  color=(255, 0, 0),
                                                  scale=10)

        # Hitung sudut antara landmark 11, 13, dan 15
        angle, img = detector.findAngle(lmList[11][0:2],
                                        lmList[13][0:2],
                                        lmList[15][0:2],
                                        img=img,
                                        color=(0, 0, 255),
                                        scale=10)

        # Cek apakah sudut mendekati 50 derajat (offset 10)
        isCloseAngle50 = detector.angleCheck(myAngle=angle,
                                             targetAngle=50,
                                             offset=10)
        print(f"Sudut: {angle:.2f}, Dekat 50°? {isCloseAngle50}")

    # Tampilkan hasil
    cv2.imshow("Pose + Angle", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
