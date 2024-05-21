import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

# 定义一个简单的数据集
class SentenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = []
        self.labels = []

        # 从文件中读取数据
        with open(file_path, 'r') as f:
            for line in f:
                text, label = line.strip().split('\t')
                self.texts.append(text)
                self.labels.append(int(label))

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

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 加载保存的最佳模型
model.load_state_dict(torch.load('best_model.pt'))

# 将模型移动到设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备数据集和数据加载器
file_path = './datasets/test_sentence_dataset.txt'  # 测试数据集的文件路径
batch_size = 4  # 指定批大小
test_dataset = SentenceDataset(file_path, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 测试模型
model.eval()
correct_predictions = 0
total_predictions = 0
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        correct_predictions += torch.sum(predictions == labels)
        total_predictions += labels.size(0)

accuracy = correct_predictions.double() / total_predictions
print(f"Test Accuracy: {accuracy:.4f}")
