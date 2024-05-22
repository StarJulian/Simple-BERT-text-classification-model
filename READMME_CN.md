
# 简单的 BERT 文本分类模型

本仓库提供了一个使用 BERT（双向编码器表示）进行文本分类的简单实现。

## 特点

- 使用 Hugging Face 的 Transformers 库。
- 实现简单易懂。
- 包含数据预处理、模型训练和预测的脚本。

## 环境要求

- Python 3.6+
- PyTorch
- Transformers
- scikit-learn
- pandas

使用以下命令安装所需包：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 克隆仓库
```bash
git clone https://github.com/StarJulian/Simple-BERT-text-classification-model.git
cd Simple-BERT-text-classification-model
```

### 2. 准备数据集
将您的数据集准备成包含 `text` 和 `label` 列的 txt 格式。

### 3. 训练模型
```bash
python train.py 
```
### 4. 评估模型
```bash
python test.py 
```

### 5. 进行预测
```bash
python predict.py 
```

## 仓库结构


```
Simple-BERT-text-classification-model/
├── datasets/
│   ├── your_train_dataset.txt
│   ├── your_testt_dataset.txt
├── saved_models/
│   └── your_model.pth
├── datasets.py
├── main.py
├── predict.py
├── requirements.txt
├── test.py
├── train.py
└── README.md
```

## 脚本说明

- `datasets.py`: 数据预处理脚本。
- `train.py`: 训练 BERT 模型的脚本。
- `test.py`: 评估模型的脚本。
- `predict.py`: 对新数据进行预测的脚本。

## 许可证

本项目使用 Apache-2.0 许可证。详细信息请参阅 [LICENSE](LICENSE) 文件。

## 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)

## 联系方式

如果有问题或需要帮助，请在仓库中创建一个 issue。
```

你可以根据具体的项目需求更新路径、文件名和指令。