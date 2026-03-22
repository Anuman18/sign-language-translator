import cv2
import csv
from detector import HandDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    label = input("Enter label (A, B, C...): ")

    with open("data/dataset.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        print("Press 's' to save data, 'q' to quit")

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame, landmarks = detector.detect_hands(frame)

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1)

            if key == ord("s") and landmarks:
                row = landmarks + [label]
                writer.writerow(row)
                print(f"Saved sample for {label}")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()