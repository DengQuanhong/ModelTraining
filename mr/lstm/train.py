from mdataset import get_dataloader
from model import LSTMClassifier
from config import my_config
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from tqdm import tqdm


def run_epoch(model, train_iterator, optimzer, loss_fn,device,epoch,writer:SummaryWriter,config):  # 训练模型
    '''
    :param model:模型
    :param train_iterator:训练数据的迭代器
    :param dev_iterator: 验证数据的迭代器
    :param optimzer: 优化器
    :param loss_fn: 损失函数
    '''
    total_step = 0
    for i, batch in tqdm(enumerate(train_iterator),desc="Epoch {}/{}".format(epoch+1, config.epochs)):
        total_step += 1
        input,label = batch[0],batch[1]
        input = input.to(device)
        label = label.to(device)
        pred = model(input)  # 预测
        loss = loss_fn(pred, label)  # 计算损失值
        
        optimzer.zero_grad()
        loss.backward()  # 误差反向传播
        optimzer.step()  # 优化一次
        
        lr = get_lr(i,config)
        for param_group in optimzer.param_groups:
            param_group['lr'] = lr

        if i % 30 == 0:  # 训练30个batch后查看损失值和准确率
            writer.add_scalar("lstmtrain/loss",loss.detach().cpu().item(),global_step=epoch*len(train_iterator)+i)
            writer.flush()


def evaluate_model(model, dev_iterator,device):  # 评价模型
    '''
    :param model:模型
    :param dev_iterator:待评价的数据
    :return:评价（准确率）
    '''
    all_pred = []
    all_y = []
    total_step = 0
    for i, batch in tqdm(enumerate(dev_iterator),total=len(dev_iterator),desc="Eval"):
        total_step+=1
        input = batch[0].to(device)
        label = batch[1]

        y_pred = model(input)  # 预测
        predicted = torch.argmax(y_pred,dim=-1) # 选择概率最大作为当前数据预测结果
        all_pred.extend(predicted.detach().cpu().numpy())
        all_y.extend(label.numpy())
    score = accuracy_score(all_y, np.array(all_pred).flatten())  # 计算准确率
    return score
# 定义学习率调整函数
def get_lr(step,config):
    if step < config.warm_up_steps:
        return config.lr * (step / config.warm_up_steps)
    else:
        return config.lr
# 定义早停函数
def early_stop(val_accs, patience):
    if len(val_accs) < patience:
        return False
    for i in range(1, patience+1):
        if val_accs[-i] > val_accs[-i-1]:
            return False
    return True

if __name__ == '__main__':
    config = my_config()  # 配置对象实例化
    
    train_loader,val_loader,test_loader = get_dataloader(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 初始化模型
    model = LSTMClassifier(my_config)
    model = model.to(device)
    writer = SummaryWriter("/data/xuzhi/code/ad/hard-label-attack/ensemble/mr/logs/lstm")

    optimzer = torch.optim.Adam(model.parameters(), lr=config.lr)  # 优化器
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    best_acc = 0.0
    val_accs = []
    for i in range(config.epochs):
        print(f'epoch:{i + 1}')
        run_epoch(model, train_loader, optimzer, loss_fn,device,i,writer,config)

        # 训练一次后评估一下模型
        val_acc = evaluate_model(model, val_loader,device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, "model.pt")
        val_accs.append(val_acc)
        print(val_acc)
        if early_stop(val_accs, config.patience):
            print("Early stopping.")
            break

        if early_stop(val_accs, config.patience):
            break

        writer.add_scalar("lstm/acc",val_acc,global_step=i,walltime=None)
        writer.flush()
    
    test_acc = evaluate_model(model,test_loader,device)
    print(test_acc)



