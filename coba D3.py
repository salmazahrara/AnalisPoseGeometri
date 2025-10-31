import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import time

# ===== STEP 1: Inisialisasi =====
detector = FaceMeshDetector(maxFaces=1)
cap = cv2.VideoCapture(0)

# Indeks landmark untuk mata kiri dan kanan
LEFT_EYE = [33, 133, 159, 145]  # kanan, kiri, atas, bawah
RIGHT_EYE = [362, 263, 386, 374]

def eye_aspect_ratio(face, eye_points):
    right, left, top, bottom = eye_points
    horizontal = np.linalg.norm(np.array(face[right]) - np.array(face[left]))
    vertical = np.linalg.norm(np.array(face[top]) - np.array(face[bottom]))
    return vertical / (horizontal + 1e-6)

# ===== STEP 2: Variabel kalibrasi =====
calibration_time = 5   # detik
blink_threshold = None
start_time = time.time()
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 2  # jumlah frame mata tertutup berturut-turut dianggap kedip
is_closed = False

print("=== Kalibrasi dimulai... ===")
print("Hadapkan wajah ke kamera dengan mata terbuka normal")

ear_values = []  # untuk kumpulkan EAR mata terbuka

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]
        leftEAR = eye_aspect_ratio(face, LEFT_EYE)
        rightEAR = eye_aspect_ratio(face, RIGHT_EYE)
        ear_avg = (leftEAR + rightEAR) / 2

        current_time = time.time()

        # ===== STEP 3: Mode Kalibrasi =====
        if blink_threshold is None:
            ear_values.append(ear_avg)
            cv2.putText(img, "CALIBRATING... Keep eyes OPEN", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if current_time - start_time >= calibration_time:
                open_ear = np.mean(ear_values)
                blink_threshold = open_ear * 0.75  # threshold diturunkan 25% dari kondisi terbuka
                print(f"Kalibrasi selesai! EAR rata-rata terbuka: {open_ear:.3f}")
                print(f"Threshold ditetapkan: {blink_threshold:.3f}")
                print("Silahkan kedip untuk mendeteksi...")
        else:
            # ===== STEP 4: Mode Deteksi =====
            cv2.putText(img, f"EAR Avg: {ear_avg:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(img, f"Threshold: {blink_threshold:.3f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            if ear_avg < blink_threshold:
                closed_frames += 1
                if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                    blink_count += 1
                    is_closed = True
            else:
                closed_frames = 0
                is_closed = False

            cv2.putText(img, f"Blink Count: {blink_count}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Blink Detection with Auto Calibration", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program dihentikan.")
