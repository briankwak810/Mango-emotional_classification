from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import torch

MODEL_NAME = "beomi/KcELECTRA-base"

# 1. Load Dataset
df = pd.read_csv("data/mango_emotional_classification.csv", encoding="cp949")
dataset = Dataset.from_pandas(df)

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# 3. Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 4. Training Args
training_args = TrainingArguments(
    output_dir="./classifier_checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 5. Metrics
import evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return accuracy.compute(predictions=preds, references=labels)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# 7. Save
model.save_pretrained("./mango-recall-classifier")
tokenizer.save_pretrained("./mango-recall-classifier")
