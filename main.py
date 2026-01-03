import cv2
import time
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "hand_landmarker.task"),
    running_mode = VisionRunningMode.IMAGE,
    num_hands = 1
)

detector = HandLandmarker.create_from_options(options)

#hand connection
hand_connection = [
    (0, 1), (1, 2), (2, 3), (3, 4),                 # Thumb
    (2, 5), (5, 9), (9, 13), (13, 17), (17, 0),     # Wist
    (5, 6), (6, 7), (7, 8),                         # Index
    (9, 10), (10, 11), (11, 12),                    # Middle
    (13, 14), (14, 15), (15, 16),                   # Ring
    (17, 18), (18, 19), (19, 20)                    # Pinky
]

p_time = 0

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    
    if not success:
        break
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(mp.ImageFormat.SRGB, rgb)
    
    result = detector.detect(mp_img)
    
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            
            # h -> height
            # w -> width
            # _ -> nunmber of channels
            h, w, _ = img.shape
            lm_list = []
            
            # Converts landmarks to pixel
            for lm in hand:
                lm_list.append((int(lm.x*w), int(lm.y*h)))
            
            # Join the lines of hand landmarks
            for start, end in hand_connection:
                cv2.line(img, 
                        lm_list[start], 
                        lm_list[end], 
                        (0, 255, 0), 
                        2
                        )
                
            # Converts circles to landmarks
            for x,y in lm_list:
                cv2.circle(img,
                            (x, y),
                            5, 
                            (0, 0, 255), 
                            -1
                            )
            
            # At thumb and index adding color to specify the points
            x1, y1 = lm_list[4][0] , lm_list[4][1]
            x2, y2 = lm_list[8][0] , lm_list[8][1]
            
            cv2.circle(img,
                        (x1, y1),
                        10,
                        (255, 0, 255),
                        cv2.FILLED
                        )
            cv2.circle(img,
                        (x2, y2),
                        10,
                        (255, 0, 255),
                        cv2.FILLED
                        )
            
            # Now join these two points
            cv2.line(img,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    5
                    )
            
            # Center point of the line
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            cv2.circle(img,
                        (cx, cy),
                        10,
                        (255, 0, 255),
                        cv2.FILLED
                        )
            
            
            
    # Show FPS on the screen
    cv2.putText(img,
                f'FPS: {int(fps)}',
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 255),
                2
                )

    cv2.imshow("Volume Control ML Model", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows