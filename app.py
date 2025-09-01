import os
from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

# Use smaller model for Render free tier
try:
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

def calculate_perplexity(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return 1000

def convert_perplexity_to_score(ppl):
    score = max(0, min(100, (ppl - 10) * 3))
    return round(score, 1)

@app.route('/')
def index():
    return jsonify({"status": "AI Detection API (distilgpt2) is live"})

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text or len(text) < 20:
            return jsonify({"error": "Text too short or missing."}), 400

        ppl = calculate_perplexity(text)
        score = convert_perplexity_to_score(ppl)
        return jsonify({"ai_score": score, "perplexity": round(ppl, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
