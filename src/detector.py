import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        """
        Detect hands and draw landmarks
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        landmark_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y, lm.z))

        return frame, landmark_list