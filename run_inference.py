from inference_module import InferenceModule

def main():
    model_path = "model/checkpoint-376"
    inference = InferenceModule(model_path)

    texts = ["I love this movie", "This movie sucks!"]
    results = inference.predict(texts)
    print(results)

if __name__ == "__main__":
    main()
