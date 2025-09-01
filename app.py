from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

# Load GPT2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def convert_perplexity_to_score(ppl):
    # Normalize perplexity to a 0–100 scale where lower PPL = more human
    score = max(0, min(100, (ppl - 10) * 3))
    return round(score, 1)

@app.route('/')
def home():
    return jsonify({"message": "AI Detection API is live."})

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text or len(text) < 20:
        return jsonify({"error": "Text is too short or missing."}), 400

    ppl = calculate_perplexity(text)
    ai_score = convert_perplexity_to_score(ppl)

    return jsonify({"ai_score": ai_score, "perplexity": round(ppl, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
