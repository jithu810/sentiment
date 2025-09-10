class Config:
    # Model
    model_name = "distilbert-base-uncased"
    num_labels = 3

    # Dataset
    dataset_name = "sreejith8100/Sp1786_multiclass-sentiment-analysis-dataset"
    train_size = None
    eval_size = None
    test_size = None

    # Training
    output_dir = "test1"
    learning_rate = 2e-5
    batch_size = 16
    num_epochs = 3
    weight_decay = 0.01
    logging_steps = 100

    # HuggingFace Hub
    push_to_hub = True
    hub_token = ""  # keep in .env in real case
    repo_id = "sreejith8100/test1" 
