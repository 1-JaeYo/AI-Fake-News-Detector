from flask import Flask, request, jsonify, render_template
from model import load_model, predict_news

app = Flask(__name__)

# Load model and tokenizer
model, tokenizer = load_model()


@app.route('/')
def index():
    return render_template('index.html')  # Front-end form


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    prediction = predict_news(text, model, tokenizer)
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
