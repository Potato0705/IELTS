# scorer/infer_scorer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "scorer/model"  # 你刚刚保存的路径

class LocalIeltsScorer:
    def __init__(self, model_dir: str = MODEL_DIR, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, essay: str) -> float:
        inputs = self.tokenizer(
            essay,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze().item()
        # 如果你训练时 band 就是 0-9 浮点就不用再变换
        return float(logits)

# 简单测试用
if __name__ == "__main__":
    scorer = LocalIeltsScorer()
    demo_essay = "Nowadays, many people believe that..."
    pred = scorer.score(demo_essay)
    print("Predicted band:", pred)
