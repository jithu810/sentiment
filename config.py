class Config:
    # Model
    model_name = "google/muril-base-cased"
    num_labels = 3

    # Dataset
    dataset_name = "sreejith8100/indic_sentiment_analyzer_cleaned"
    train_size = None
    eval_size = None
    test_size = None

    # Training
    output_dir = "indian_output"
    learning_rate = 2e-5
    batch_size = 16
    num_epochs = 3
    weight_decay = 0.01
    logging_steps = 100

    # HuggingFace Hub
    push_to_hub = True
    hub_token = ""  # keep in .env in real case
    repo_id = "sreejith8100/indiantest2" 
