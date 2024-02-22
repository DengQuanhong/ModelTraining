from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("/data/xuzhi/models/bert-base-uncased/")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load AG_News dataset
dataset = load_from_disk("/data/xuzhi/datasets/ag_news_dataset")
# 自定义数据预处理函数
def preprocess_function(example):
   return tokenizer(
        example["text"], truncation=True,padding='max_length',max_length=328
    )
    
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# 对训练集和验证集进行预处理
train_dataset = train_dataset.map(preprocess_function, batched=True,num_proc=20)
test_dataset = test_dataset.map(preprocess_function, batched=True,num_proc=20)

train_dataset = train_dataset.remove_columns(['text','token_type_ids','attention_mask'])
test_dataset=test_dataset.remove_columns(['text','token_type_ids','attention_mask'])


# 划分训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


# 定义批处理大小和并行加载的工作数
batch_size = 512
num_workers = 20
def collate_fn(batch):
    labels = []
    input_ids = []
    for ite in batch:
        labels.append(ite['label'])
        input_ids.append(ite['input_ids'])
    return torch.tensor(input_ids),torch.tensor(labels)
# 创建DataLoader对象
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn, num_workers=num_workers)


import torch
import torch.nn as nn
import pdb
# 自定义TextCNN模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,dropout=0.3,bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, input_ids,**kwargs):        
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        output = torch.sum(output,dim=1)
        output = self.fc(output)
        return output
    

import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

num_epochs = 50
learning_rate = 1e-3
warm_up_steps = 100
max_steps = len(train_dataset) // batch_size * num_epochs
patience = 6

# 初始化模型
input_size = len(tokenizer)
hidden_size = 256
output_size = 4

model = LSTMClassifier(len(tokenizer), hidden_size,output_size).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 定义学习率调整函数
def get_lr(step):
    if step < warm_up_steps:
        return learning_rate * (step / warm_up_steps)
    else:
        return learning_rate

# 定义早停函数
def early_stop(val_loss, patience):
    if len(val_loss) < patience:
        return False
    for i in range(1, patience+1):
        if val_loss[-i] < val_loss[-i-1]:
            return False
    return True

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# 训练模型
train_loss = 0.0
val_loss = []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    total_steps = 0
    
    for batch in tqdm(train_dataloader, desc="Epoch {}/{}".format(epoch+1, num_epochs)):
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)
        
        # 前向传播
        logits = model(input_ids)
        loss = criterion(logits, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        total_steps += 1

        # 更新学习率
        lr = get_lr(total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        
    model.eval()
    val_epoch_loss = 0.0
    val_predictions = []
    val_labels_ = []
    
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_input_ids = val_batch[0].to(device)
            val_labels = val_batch[1].to(device)
            
            val_logits = model(val_input_ids)
            val_batch_loss = criterion(val_logits, val_labels)
            
            val_epoch_loss += val_batch_loss.item()
            
            _, predicted_labels = torch.max(val_logits, 1)
            val_predictions.extend(predicted_labels.tolist())
            val_labels_.extend(val_labels.tolist())

    val_accuracy = accuracy_score(val_labels_, val_predictions)
    val_loss.append(val_epoch_loss / len(val_dataloader))
    print(val_accuracy)

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        model.to("cpu")
        torch.save(model, "model.pt")
        model.to(device)
    
    if early_stop(val_loss, patience):
        print("Early stopping.")
        break

    if early_stop(val_loss, patience):
        break

# # 加载最佳模型参数
model= torch.load("model.pt")

# 在测试集上评估模型
model.eval()
test_predictions = []
test_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)

        logits = model(input_ids)
        _, predicted_labels = torch.max(logits, 1)

        test_predictions.extend(predicted_labels.tolist())
        test_labels.extend(labels.tolist())

test_accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))