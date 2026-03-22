import cv2
from detector import HandDetector
from model import GestureModel


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    model = GestureModel()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, landmarks = detector.detect_hands(frame)

        prediction = None
        if landmarks:
            prediction = model.predict(landmarks)

        # Display prediction
        if prediction:
            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()