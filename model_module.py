import torch
from transformers import AutoModelForSequenceClassification
from config import Config
from logger import get_logger

logger = get_logger(__name__)

class ModelModule:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        logger.info(f"Loading model: {self.config.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=self.config.num_labels
        )
        return model.to(self.device)
