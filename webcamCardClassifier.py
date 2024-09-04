import cv2
from cardClassifier import CardClassifier

class WebcamCardClassifier(CardClassifier):
    def __init__(self, model_path, threshold=0.5, camNum=0):
        super().__init__(model_path, threshold)
        self.camNum = camNum

    def predict_from_webcam(self):
        cap = cv2.VideoCapture(self.camNum)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = self.card_prediction_from_video(frame)
            cv2.imshow('Card Prediction', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def card_prediction_from_video(self, frame):
        results = self._run_model(frame)
        annotated_frame = self._annotate_predictions(frame, results)
        return annotated_frame

    def record_video(self, save_path):
        cap = cv2.VideoCapture(self.camNum)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_path, fourcc, 20.0, (640, 480))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = self.card_prediction_from_video(frame)
            out.write(annotated_frame)  # Write the frame to the video file
            cv2.imshow('Card Prediction', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release everything when job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
