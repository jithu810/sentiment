from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from config import Config
from logger import get_logger

logger = get_logger(__name__)

class DataModule:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def load_data(self):
        logger.info(f"📥 Loading dataset: {self.config.dataset_name} ...")
        dataset = load_dataset(self.config.dataset_name)

        train_ds = dataset["train"]
        eval_ds = dataset["validation"]
        test_ds = dataset["test"]

        logger.info(
            f"✅ Dataset loaded: Train={len(train_ds)}, "
            f"Validation={len(eval_ds)}, Test={len(test_ds)}"
        )
        return train_ds, eval_ds, test_ds

    def preprocess(self, examples):
        tokenized = self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
        tokenized["labels"] = examples["labels"]  # already correct in hub dataset
        return tokenized

    def prepare(self):
        train, eval, test = self.load_data()

        logger.info("⚡ Tokenizing datasets...")

        tokenized_train = train.map(
            self.preprocess,
            batched=True,
            remove_columns=["text"]  # only drop text, keep labels
        )
        tokenized_eval = eval.map(
            self.preprocess,
            batched=True,
            remove_columns=["text"]
        )
        tokenized_test = test.map(
            self.preprocess,
            batched=True,
            remove_columns=["text"]
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        return tokenized_train, tokenized_eval, tokenized_test, data_collator
