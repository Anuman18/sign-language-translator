import cv2
from detector import HandDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, landmarks = detector.detect_hands(frame)

        # Debug: print number of landmarks
        if landmarks:
            print(f"Detected {len(landmarks)} landmarks")

        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()