import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from collections import deque

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.6)

# Stabilizer (agar tidak kedip-kedip)
gesture_history = deque(maxlen=8)
last_gesture = "..."

def get_finger_states(lmList):
    """
    Mengembalikan status jari: 1 = terbuka, 0 = tertutup
    urutan: [Thumb, Index, Middle, Ring, Pinky]
    """
    finger_states = []

    # Thumb (lebih horizontal dibanding jari lain)
    # Cek apakah ujung ibu jari lebih ke kanan/kiri dari joint-nya
    if lmList[4][0] > lmList[3][0]:  # untuk tangan kanan (flipType=True)
        finger_states.append(1)
    else:
        finger_states.append(0)

    # Jari lain (vertikal: tip lebih tinggi dari PIP joint)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if lmList[tip][1] < lmList[pip][1]:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states

def classify_gesture(finger_states, lmList):
    thumb, index, middle, ring, pinky = finger_states

    # ROCK: semua jari tertutup
    if finger_states == [0, 0, 0, 0, 0]:
        return "ROCK"

    # PAPER: semua jari terbuka
    if finger_states == [1, 1, 1, 1, 1]:
        return "PAPER"

    # SCISSORS: telunjuk & tengah terbuka, lainnya tertutup
    if finger_states == [0, 1, 1, 0, 0]:
        return "SCISSORS"

    # THUMBS UP: ibu jari terbuka ke atas + jari lain tertutup
    wrist = np.array(lmList[0][:2])
    thumb_tip = np.array(lmList[4][:2])
    if thumb == 1 and index == middle == ring == pinky == 0:
        if thumb_tip[1] < wrist[1] - 40:  # ibu jari lebih tinggi dari pergelangan
            return "THUMBS_UP"

    # OK gesture: ujung ibu jari & telunjuk saling menyentuh (lingkaran)
    thumb_tip = np.array(lmList[4][:2])
    index_tip = np.array(lmList[8][:2])
    if np.linalg.norm(thumb_tip - index_tip) < 30:
        return "OK"

    return "UNKNOWN"


while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img, flipType=True)  # kamera depan

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        finger_states = get_finger_states(lmList)
        gesture = classify_gesture(finger_states, lmList)
        gesture_history.append(gesture)

        # Tentukan gesture stabil
        if gesture_history.count(gesture) > len(gesture_history) // 2:
            last_gesture = gesture

        # Warna box tergantung gesture
        colors = {
            "ROCK": (0, 0, 255),
            "PAPER": (255, 255, 255),
            "SCISSORS": (0, 255, 0),
            "THUMBS_UP": (255, 255, 0),
            "OK": (0, 255, 255),
            "UNKNOWN": (128, 128, 128)
        }
        color = colors.get(last_gesture, (255, 255, 255))

        x, y, w, h = hand["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, f"{last_gesture}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Hand Gesture Detection - Akurat Mode", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
