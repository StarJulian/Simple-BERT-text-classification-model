import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import os

# 定义数据集类
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_v = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text_v,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'text': text_v,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 读取数据文件的函数
def read_dataset(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().split('\t')  # 假设数据是制表符分隔
            texts.append(text)
            labels.append(int(label))  # 将标签转换为整数
    return texts, labels

# 准备数据文件路径
train_file_path = './datasets/train_sentence_dataset.txt'
val_file_path = './datasets/test_sentence_dataset.txt'

# 读取训练数据和验证数据
train_texts, train_labels = read_dataset(train_file_path)
val_texts, val_labels = read_dataset(val_file_path)

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 创建数据集实例
train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)

# 准备数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=16)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# 定义优化器、调度器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs_num = 100  # 训练的epoch数量
num_training_steps = len(train_dataloader) * epochs_num
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# 检查输出目录是否存在
output_dir = './saved_model/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"目录 {output_dir} 已创建。")
else:
    print(f"目录 {output_dir} 已存在。")

# 初始化最佳损失为正无穷
best_loss = float('inf')

# 训练模型
model.train()

# 将模型移动到设备上
model.to(device)

# 初始化最佳损失和最佳准确率
best_loss = float('inf')
best_accuracy = 0.0

# 训练模型
model.train()
for epoch in range(epochs_num):
    # ...（省略训练代码）
    epoch_loss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}
        
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    # 进行验证
    model.eval()
    epoch_val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()if k != 'text'}
            outputs = model(**batch)
            val_loss = loss_fn(outputs.logits, batch['labels'])
            epoch_val_loss += val_loss.item()
            
            _, predicted_labels = torch.max(outputs.logits, 1)
            total_predictions += batch['labels'].size(0)
            correct_predictions += (predicted_labels == batch['labels']).sum().item()

    epoch_val_loss /= len(val_dataloader)
    val_accuracy = correct_predictions / total_predictions

    print(f"Epoch {epoch + 1}/{epochs_num}, Train Loss: {epoch_loss / len(train_dataloader)}, "
          f"Validation Loss: {epoch_val_loss}, Validation Accuracy: {val_accuracy}")

    # 检查是否是最佳模型
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model_path = os.path.join(output_dir, 'best_text2cls.pth'.format(epoch + 1))
        torch.save(model.state_dict(), best_model_path)
        print(f"找到最佳模型，验证准确率为 {best_accuracy:.4f}，在Epoch {epoch + 1}，模型已保存。")

# 保存最终模型
final_model_path = os.path.join(output_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print("Training complete. Final model has been saved to", final_model_path)