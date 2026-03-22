import cv2
from detector import HandDetector
from model import GestureModel
from word_builder import WordBuilder


def main():
    cap = cv2.VideoCapture(0)

    detector = HandDetector()
    model = GestureModel()
    word_builder = WordBuilder()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, landmarks = detector.detect_hands(frame)

        prediction = None
        if landmarks:
            prediction = model.predict(landmarks)

        # Update word
        word = word_builder.update(prediction)
        corrected = word_builder.get_corrected_word()

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

        # Display word
        cv2.putText(
            frame,
            f"Word: {word}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # Display corrected word
        cv2.putText(
            frame,
            f"Corrected: {corrected}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Sign Language Translator", frame)

        key = cv2.waitKey(1)

        # Press 'r' to reset word
        if key == ord("r"):
            word_builder.reset()

        # Press ESC to exit
        if key & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()