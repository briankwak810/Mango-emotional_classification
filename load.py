from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "./mango-recall-classifier"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # Inference mode

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_label].item()
    return predicted_label, confidence

text = "날씨가 참 좋군."
label, confidence = classify(text)
print(f"예측 라벨: {label}, 확신도: {confidence:.2f}")