from config import Config
from data_module import DataModule
from model_module import ModelModule
from trainer_module import TrainerModule
from upload_results import save_readme
from huggingface_hub import login, create_repo


def main():
    print("🔧 Loading configuration...")
    cfg = Config()

    print("📂 Preparing data...")
    data_module = DataModule(cfg)
    train_ds, eval_ds, test_ds, data_collator = data_module.prepare()
    print(f"✅ Train size: {len(train_ds)}, Eval size: {len(eval_ds)}, Test size: {len(test_ds)}")

    print("🤖 Loading model...")
    model_module = ModelModule(cfg)
    model = model_module.load_model()
    print(f"✅ Model loaded on device: {model.device}")

    print("🎯 Setting up Trainer...")
    trainer_module = TrainerModule(model, data_module.tokenizer, cfg)
    trainer = trainer_module.get_trainer(train_ds, eval_ds, data_collator)

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training finished")

    print("📊 Evaluating model on validation set...")
    metrics = trainer.evaluate()
    print(f"✅ Evaluation results: {metrics}")

    
    print("🧪 Running inference on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    print(f"✅ Test results: {test_metrics}")
    print("🏁 Training pipeline complete.")

    # 🔥 Save README with results
    save_readme(cfg, metrics, test_metrics)

    if cfg.push_to_hub:
        print("🔑 Logging into Hugging Face Hub...")
        login(token=cfg.hub_token)
        print("⬆️ Preparing Hugging Face Hub repo...")
        create_repo(cfg.repo_id, exist_ok=True)  # auto-create if missing
        print("⬆️ Pushing model to Hugging Face Hub...")
        trainer.push_to_hub(cfg.repo_id)
        print("✅ Model pushed to hub")

if __name__ == "__main__":
    main()
