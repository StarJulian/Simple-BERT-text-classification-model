import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# 定义一个简单的数据集
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

# 准备数据
epochs_num = 10
best_loss = float('inf')
output_dir = './saved_model/'

texts = ["This is a positive statement.", "This is a negative statement."]
labels = [1, 0]  # 假设1为肯定句，0为否定句

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

dataset = SentenceDataset(texts, labels, tokenizer)

# 准备数据加载器
dataloader = DataLoader(dataset, batch_size=2)

# 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(dataloader) *  epochs_num # 假设我们训练3个epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# 将模型和分词器移动到设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



model.train()
for epoch in range(3):  # 简单的3个epoch示例
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()  # 这个调用通常也是必要的，特别是当你使用学习率调度器时
        optimizer.zero_grad()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

print("Training complete.")
