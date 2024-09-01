import os
from countCards import CountCards

#turn this on to train new model 
TrainNewModel = False
cwd = os.getcwd()

if __name__ == "__main__": 
    model_path = f"{cwd}\\playing-card-classifier\\weights\\best.pt"
    if TrainNewModel:
        trainer = CardModelTrainer(
            model_config="yolov8n.yaml",
            data_yaml_file="data\\data.yaml",
            project=f"{cwd}",
            experiment="playing-card-classifier",
            batch_size=32,
            epochs=50,
            device=0,
            patience=5,
            img_size=640,
            verbose=True
        )
        
        trainer.train_model()
    #run predicitons from webcam 
    # webcam_classifier = WebcamCardClassifier(model_path,camNum=1)
    count_cards = CountCards(model_path, numDecks=1)
    count_cards.predict_from_webcam()
