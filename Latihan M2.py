import cv2
import mediapipe as mp
mpose = mp.solutions.pose
pose = mpose.Pose()
mdraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        break

    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hasil = pose.process(imgrgb)
    if hasil.pose_landmarks:
        mdraw.draw_landmarks(img, hasil.pose_landmarks, mpose.POSE_CONNECTIONS)
        lm = hasil.pose_landmarks.landmark
        pundak_kiri = lm[mpose.PoseLandmark.LEFT_SHOULDER.value]
        tangan_kiri = lm[mpose.PoseLandmark.LEFT_WRIST.value]
        pundak_kanan = lm[mpose.PoseLandmark.RIGHT_SHOULDER.value]
        tangan_kanan = lm[mpose.PoseLandmark.RIGHT_WRIST.value]

        if tangan_kiri.y < pundak_kiri.y:
            cv2.putText(img, "Tangan Kiri Terangkat",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        if tangan_kanan.y < pundak_kanan.y:
            cv2.putText(img, "Tangan Kanan Terangkat",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

    cv2.imshow("Deteksi Tangan", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
