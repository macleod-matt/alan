from ultralytics import YOLO

class CardModelTrainer:
    def __init__(self, model_config, data_yaml_file, project, experiment, batch_size=32, epochs=50, device=0, patience=5, img_size=640, verbose=True):
        """
        Initialize the CardModelTrainer with training configurations.

        :param model_config: Path to the model configuration file (e.g., 'yolov8n.yaml').
        :param data_yaml_file: Path to the data YAML file.
        :param project: Path to the project directory.
        :param experiment: Name of the experiment.
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs for training.
        :param device: Device to use for training (e.g., 0 for GPU).
        :param patience: Number of epochs to wait for improvement before early stopping.
        :param img_size: Size of the images for training.
        :param verbose: Whether to print detailed logs.
        """
        self.model_config = model_config
        self.data_yaml_file = data_yaml_file
        self.project = project
        self.experiment = experiment
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.patience = patience
        self.img_size = img_size
        self.verbose = verbose

    def train_model(self):
        """Method to start training the model with the specified parameters."""
        model = YOLO(self.model_config)
        results = model.train(data=self.data_yaml_file,
                              epochs=self.epochs,
                              project=self.project,
                              name=self.experiment,
                              batch=self.batch_size,
                              device=self.device,
                              patience=self.patience,
                              imgsz=self.img_size,
                              verbose=self.verbose,
                              val=True)
        return results
