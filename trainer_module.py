import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback
from logger import get_logger


logger = get_logger(__name__)

import wandb
wandb.init(mode="disabled")

class SamplesProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        samples_done = state.global_step * args.per_device_train_batch_size
        print(f"Samples processed: {samples_done}")

class TrainerModule:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        
        accuracy = self.accuracy_metric.compute(
            predictions=preds, references=labels
        )["accuracy"]

        f1 = self.f1_metric.compute(
            predictions=preds, references=labels, average="weighted"
        )["f1"]
        return {"accuracy": accuracy, "f1": f1}

    def get_trainer(self, train_ds, eval_ds, data_collator):
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,             # keep only last 2 checkpoints
            load_best_model_at_end=True,   # load best checkpoint when training ends
            metric_for_best_model="f1",    # choose best by f1 (or "accuracy")
            greater_is_better=True,        # higher metric = better
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            push_to_hub=self.config.push_to_hub,

        )

        return Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            # callbacks=[SamplesProgressCallback()]
        )
