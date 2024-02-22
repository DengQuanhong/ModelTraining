from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch
import os
import argparse
from datasets import load_from_disk
import warnings

warnings.filterwarnings("ignore",category=UserWarning)


def tokenizer_func(sample):
    return tokenizer(
        sample["text"], truncation=True,padding='max_length',max_length=128
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = predictions.argmax(axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, required=True
    )

    parser.add_argument(
        "--batch_size", type=int, required=True
    )
    parser.add_argument("--cuda_device", type=str, required=True)

    parser.add_argument("--tokenizer_path", type=str, required=True)
    
    parser.add_argument("--model_path",type=str,required=True)

    parser.add_argument("--saved_path", type=str, required=True)
    
    parser.add_argument("--lr",type=float,required=True)
    
    parser.add_argument("--log_dir",type=str,required=True)

    args = parser.parse_args()
    print("parser okk.")
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    model_name = args.model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name, num_labels=2
    # )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    print("="*50)
    print(model)
    print("="*50)
    if "xlnet" in args.model_path:
        for name,param in model.named_parameters():
            if not name.startswith("logits_proj"):
                param.requires_grad = False
    else:
        for name,param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
                
    
    model = model.to(device)
    # Load AG_News dataset
    dataset = load_from_disk(args.dataset_path)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(tokenizer_func, batched=True, num_proc=20)
    test_dataset = test_dataset.map(tokenizer_func, batched=True, num_proc=20)
    if not os.path.exists(args.saved_path):
        os.mkdir(args.saved_path)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    training_args = TrainingArguments(
        output_dir=args.saved_path,
        logging_dir = args.log_dir,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=300,
        weight_decay=0.01,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=50)]
        
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    print(eval_results)

    
"""
python train_classifier.py \
    --batch_size 64 \
    --dataset_path /data/xuzhi/datasets/rotten_tomatoes_dataset \
    --cuda_device 1 \
    --tokenizer_path /data/xuzhi/models/bert-base-uncased \
    --model_path /data/xuzhi/models/bert-base-uncased \
    --lr 2e-5 \
    --saved_path ./models/bert-base-uncased\
    --log_dir ./logs/bert-base-uncased
    
    
python train_classifier.py \
    --batch_size 64 \
    --dataset_path /data/xuzhi/datasets/rotten_tomatoes_dataset \
    --cuda_device 1 \
    --tokenizer_path /data/xuzhi/models/albert-base-v2 \
    --model_path /data/xuzhi/models/albert-base-v2 \
    --lr 2e-5 \
    --log_dir ./logs/albert-base-v2 \
    --saved_path ./models/albert-base-v2

cd ensemble/mr &&   
python train_classifier.py \
    --batch_size 64 \
    --dataset_path /data/xuzhi/datasets/rotten_tomatoes_dataset \
    --cuda_device 1 \
    --tokenizer_path /data/xuzhi/models/distilbert-base-case \
    --model_path /data/xuzhi/models/distilbert-base-case \
    --lr 2e-5 \
    --log_dir ./logs/distilbert-base-case \
    --saved_path ./models/distilbert-base-case

python train_classifier.py \
    --batch_size 64 \
    --dataset_path /data/xuzhi/datasets/rotten_tomatoes_dataset \
    --cuda_device 1 \
    --tokenizer_path /data/xuzhi/models/roberta-base \
    --model_path /data/xuzhi/models/roberta-base \
    --lr 2e-5 \
    --log_dir ./logs/roberta-base \
    --saved_path ./models/roberta-base
    
    
python train_classifier.py \
    --batch_size 64 \
    --dataset_path /data/xuzhi/datasets/rotten_tomatoes_dataset \
    --cuda_device 1 \
    --tokenizer_path /data/xuzhi/models/xlnet-base-cased \
    --model_path /data/xuzhi/models/xlnet-base-cased \
    --lr 2e-5 \
    --log_dir ./logs/xlnet-base-cased \
    --saved_path ./models/xlnet-base-cased
    

"""