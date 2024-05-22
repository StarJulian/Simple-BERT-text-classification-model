
# Simple BERT Text Classification Model

This repository provides a simple implementation of a text classification model using BERT (Bidirectional Encoder Representations from Transformers).

## Features

- Uses Hugging Face's Transformers library.
- Simple and easy-to-understand implementation.
- Includes scripts for data preprocessing, model training, and prediction.

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- scikit-learn
- pandas

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/StarJulian/Simple-BERT-text-classification-model.git
cd Simple-BERT-text-classification-model
```

### 2. Prepare Your Dataset
Prepare your dataset in CSV format with `text` and `label` columns.


### 3. Train the Model
```bash
python train.py 
```

### 4. Evaluate the Model
```bash
python test.py 
```

### 5. Make Predictions
```bash
python predict.py
```

## Repository Structure

```
Simple-BERT-text-classification-model/
├── data/
│   ├── your_dataset.csv
│   ├── processed_dataset.csv
├── models/
│   └── bert_model.pth
├── datasets.py
├── main.py
├── predict.py
├── requirements.txt
├── test.py
├── train.py
└── README.md
```

## Scripts

- `datasets.py`: Script for data preprocessing.
- `train.py`: Script for training the BERT model.
- `test.py`: Script for evaluating the model.
- `predict.py`: Script for making predictions on new data.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)