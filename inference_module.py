from transformers import pipeline
from logger import get_logger

logger = get_logger(__name__)

class InferenceModule:
    def __init__(self, model_path):
        logger.info(f"Loading inference model from {model_path}")
        self.pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

    def predict(self, texts):
        return self.pipeline(texts)
