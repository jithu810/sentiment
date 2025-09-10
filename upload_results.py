import os

def save_readme(cfg, metrics, test_metrics):
    readme_content = f"""
    # DistilBERT for Multiclass Sentiment Analysis  

    ## 📖 Model Description  
    This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) for **multiclass sentiment analysis** (positive, negative, neutral).  
    It was trained using the [Sp1786/multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset).  

    - **Base Model**: {cfg.model_name}  
    - **Task**: Text Classification (Sentiment Analysis)  
    - **Labels**:  
      - `0`: Negative  
      - `1`: Neutral  
      - `2`: Positive  

    ---

    ## 📊 Dataset  
    - **Source**: {cfg.dataset_name}  
    - **Train size**: {cfg.train_size or "full"}  
    - **Eval size**: {cfg.eval_size or "split from dataset"}  
    - **Test size**: {cfg.test_size or "split from dataset"}  

    ---

    ## ⚙️ Training Setup  
    - **Optimizer**: AdamW  
    - **Learning rate**: `{cfg.learning_rate}`  
    - **Batch size**: `{cfg.batch_size}`  
    - **Epochs**: `{cfg.num_epochs}`  
    - **Weight decay**: `{cfg.weight_decay}`  

    ---

    ## 📈 Evaluation Metrics  

    | Metric        | Validation | Test |
    |---------------|------------|------|
    | Accuracy      | {metrics.get("eval_accuracy", 0):.4f} | {test_metrics.get("eval_accuracy", 0):.4f} |
    | F1 (weighted) | {metrics.get("eval_f1", 0):.4f} | {test_metrics.get("eval_f1", 0):.4f} |

    ---

    ## 🚀 Usage  

    ```python
    from transformers import pipeline

    classifier = pipeline("text-classification", model="your-username/finetuned-model")

    print(classifier("I love this product!"))  
    # [{{'label': 'POSITIVE', 'score': 0.98}}]
    ```
    """

    with open(os.path.join(cfg.output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
