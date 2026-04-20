import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
    ok = cap.isOpened()
    print(f"Camera index {i} opened = {ok}")
    cap.release()
