from webcamCardClassifier import WebcamCardClassifier
import cv2
from cardModelTrainer import CardModelTrainer


class CountCards(WebcamCardClassifier):
    def __init__(self, model_path, threshold=0.5,numDecks=1):
        super().__init__(model_path, threshold,camNum=0)
        self.runningCount = 0  # Global class variable for counting cards
        self.lastCard = None   # Tracks the last detected card
        self.consecutiveCount = 0  # Tracks the number of consecutive frames with the same card
        self.numDecks = numDecks
        suits = ["h", "d", "c", "s"]  # Hearts, Diamonds, Clubs, Spades
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        self.AvailableCards = [rank + suit for rank in ranks for suit in suits] * self.numDecks # 4 decks
    
    def update_count(self, card_name):
        if card_name[0] in ["2", "3", "4", "5", "6"] and card_name in self.AvailableCards:
            self.runningCount += 1
            self.AvailableCards.remove(card_name)
        elif card_name[0] in ["10", "J", "Q", "K", "A"] and card_name in self.AvailableCards:
            self.runningCount -= 1
            self.AvailableCards.remove(card_name)

    def check_consecutive_frames(self, card_name):
        print(f"Checking card: {card_name}")  # Debug print
        if card_name == self.lastCard:
            self.consecutiveCount += 1
        else:
            self.lastCard = card_name
            self.consecutiveCount = 1

        if self.consecutiveCount >= 5:
            if card_name in self.AvailableCards:
                self.update_count(card_name)
            self.consecutiveCount = 0  # Reset the count after updating

    def card_prediction_from_video(self, frame):
        annotated_frame = super().card_prediction_from_video(frame)

        card_name = self.extract_card_name_from_frame(annotated_frame)

        if card_name:
            self.check_consecutive_frames(card_name)

        annotated_frame = self.superimpose_running_count(annotated_frame)
        return annotated_frame

    def extract_card_name_from_frame(self, annotated_frame):
        results = self._run_model(annotated_frame)
        if results and results.boxes:
            for result in results.boxes.data.tolist():
                class_id = result[-1]  # Assuming class_id is the last item
                card_name = results.names[int(class_id)]
                return card_name
        return None

    def superimpose_running_count(self, frame):
        text = f"Running Count: {self.runningCount}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)  # Green color for the text
        thickness = 2
        position = (10, 50)
        cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        return frame
