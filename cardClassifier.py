from ultralytics import YOLO
import cv2
import os

class CardClassifier:
    def __init__(self, model_path, threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.model = YOLO(self.model_path)

    def _load_image(self, img_path):
        """Helper function to load an image from a path."""
        return cv2.imread(img_path)

    def _run_model(self, img):
        """Helper function to run the YOLO model on the image."""
        return self.model(img)[0]

    def _annotate_predictions(self, img, results):
        """Helper function to annotate the image with the model's predictions."""
        img_predict = img.copy()
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.threshold: 
                cv2.rectangle(img_predict, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(img_predict, f"{results.names[int(class_id)].upper()},{score:.2f}", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return img_predict

    def _load_annotations(self, annot_path):
        """Helper function to load ground truth annotations from a file."""
        with open(annot_path, 'r') as file: 
            lines = file.readlines()

        annotations = []
        for line in lines: 
            values = line.split()
            label = values[0]
            x, y, w, h = map(float, values[1:]) 
            annotations.append((label, x, y, w, h))
        return annotations

    def _annotate_truth(self, img, annotations, results):
        """Helper function to annotate the image with ground truth data."""
        img_truth = img.copy()
        H, W, _ = img.shape

        for ann in annotations:
            label, x, y, w, h  = ann 
            label_name = results.names[int(label)].upper()

            x1 = int((x - w / 2) * W)
            y1 = int((y - h / 2) * H)
            x2 = int((x + w / 2) * W)
            y2 = int((y + h / 2) * H)

            cv2.rectangle(img_truth, (x1, y1), (x2, y2), (200, 200, 0), 1)
            cv2.putText(img_truth, label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)

        return img_truth

    def get_annotated_frame(self, img_path, annot_path):
        """Main method to get both predicted and ground truth annotated images."""
        img = self._load_image(img_path)
        results = self._run_model(img)
        annotated_predictions = self._annotate_predictions(img, results)
        annotations = self._load_annotations(annot_path)
        annotated_truth = self._annotate_truth(img, annotations, results)

        return annotated_predictions, annotated_truth

    def _preprocess_frame(self, frame):
        """Helper function to preprocess the frame (e.g., resize, grayscale, etc.)."""
        # Resize the frame if necessary
        frame = cv2.resize(frame, (640, 480))
        # Convert to grayscale if needed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Re-orient the frame if needed (example: rotate)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def card_prediction_from_video(self, frame):
        """Method to predict and annotate a frame from a video source (webcam)."""
        # Preprocess the frame
        # processed_frame = self._preprocess_frame(frame)
        # Convert the grayscale frame back to BGR for model input
        # processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        # Run the model on the preprocessed frame
        results = self._run_model(frame)
        # Annotate the frame with predictions
        annotated_frame = self._annotate_predictions(frame, results)
        return annotated_frame