from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import os
import argparse
from datasets import load_from_disk


def tokenizer_func(sample):
    return tokenizer(
        sample["text"], truncation=True
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = predictions.argmax(axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Which dataset to attack."
    )

    parser.add_argument("--cuda_device", type=str, required=True)

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--saved_path", type=str, required=True)

    args = parser.parse_args()
    print("parser okk.")
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4
    ).to(device)

    # Load AG_News dataset
    dataset = load_from_disk(args.dataset_path)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(tokenizer_func, batched=True, num_proc=20)
    test_dataset = test_dataset.map(tokenizer_func, batched=True, num_proc=20)

    training_args = TrainingArguments(
        output_dir=args.saved_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    print(eval_results)

    
"""
python train_classifier.py \
    --dataset_path /data/xuzhi/datasets/ag_news_dataset \
    --cuda_device 1 \
    --model_path /data/xuzhi/models/models--miguelvictor--python-fromzero-lstmlm \
    --saved_path ./models/lstm
        

"""