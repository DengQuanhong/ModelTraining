from torchtext.vocab import GloVe

class my_config:
    max_length = 128  # 每句话截断长度
    batch_size = 64  # 一个batch的大小
    embedding_size = 300  # 词向量大小
    num_layers = 1  # 网络层数
    dropout = 0.5  # 遗忘程度
    output_size = 2  # 输出大小
    lr = 0.001  # 学习率
    epochs = 300  # 训练次数
    glove = GloVe(name='6B',cache="/data/xuzhi/code/ad/hard-label-attack",dim=300)
    patience = 10
    warm_up_steps = 100

