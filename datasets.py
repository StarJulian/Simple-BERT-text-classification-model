# 定义文本和标签
texts = [
    "He is wearing glasses.",
    "She is wearing glasses.",
    "He is smoking.",
    "She is smoking.",
    "He is talking on the phone.",
    "She is talking on the phone.",
    "He is not wearing glasses.",
    "She is not wearing glasses.",
    "He is not smoking.",
    "She is not smoking.",
    "He is not talking on the phone.",
    "She is not talking on the phone.",
    "There is someone in the car.",
    "There is no one in the car.",
    "He is wearing a seatbelt.",
    "He is not wearing a seatbelt."
]

labels = [
    1, 1, 1, 1, 1, 1,  # 肯定句
    0, 0, 0, 0, 0, 0,  # 否定句
    1, 1,  # 肯定句
    0, 0   # 否定句
]

# 将句子和标签写入文件
with open('sentence_dataset.txt', 'w') as f:
    for text, label in zip(texts, labels):
        f.write(f"{text}\t{label}\n")

print("File 'sentence_dataset.txt' created successfully.")
