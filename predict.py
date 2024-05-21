import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 定义一个用于预测的函数
def predict(texts, model, tokenizer, max_len=128):
    # 将模型设置为评估模式
    model.eval()
    
    # 对输入文本进行编码
    encodings = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    # 将输入数据移动到设备上
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    return predictions.cpu().numpy()

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 加载保存的最佳模型
model.load_state_dict(torch.load('best_model.pt'))

# 将模型移动到设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 示例文本
texts = [
    "He is wearing glasses.",
    "She is not smoking.",
    "There is someone in the car.",
    "The car is empty.",
    "He is wearing a seatbelt.",
    "He is not wearing a seatbelt."
]

# 进行预测
predictions = predict(texts, model, tokenizer)
print(predictions)

# 输出预测结果
for text, pred in zip(texts, predictions):
    label = 'Positive' if pred == 1 else 'Negative'
    print(f"Text: {text}\nPrediction: {label}\n")
