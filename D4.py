import cv2
from cvzone.HandTrackingModule import HandDetector

# Buka webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi detector tangan
detector = HandDetector(staticMode=False, maxHands=1,
                        modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi tangan (flipType=True agar seperti cermin)
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]  # tangan pertama yang terdeteksi
        fingers = detector.fingersUp(hand)  # output contoh: [0,1,1,0,0]
        count = sum(fingers)  # jumlah jari terbuka

        # Tampilkan informasi di layar
        cv2.putText(img, f"Fingers: {count}  {fingers}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hands + Fingers", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
