from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, methods=["POST"])

models = joblib.load('model.pkl')
# print(models)
print(f"Loaded object type: {type(models)}")
if isinstance(models, list):
    print(f"Number of models loaded: {len(models)}")

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detection</title>
    <script>
        async function getPrediction() {
            const text = document.getElementById("text").value;
            if (!text) {
                alert("Please enter some text.");
                return;
            }
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });
            const result = await response.json();
            document.getElementById("result").innerText = "Prediction: " + result.result;
        }

        function autoResizeTextarea(element) {
            element.style.height = "auto";
            element.style.height = element.scrollHeight + "px";
        }
    </script>
    <style>
        /* Reset Styles */
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          font-family: "Arial", sans-serif;
          display: flex;
          justify-content: center;
          align-items: center;
          height: 100vh;
          color: #333;
        }

        .container {
          display: flex;
          justify-content: center;
          align-items: center;
          width: 100%;
        }

        .card {
          background: #fff;
          border-radius: 10px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
          padding: 2rem;
          width: 90%;
          max-width: 500px;
          text-align: center;
          animation: fadeIn 1s ease-in-out;
        }

        h1 {
          font-size: 2rem;
          color: #4e54c8;
          margin-bottom: 0.5rem;
        }

        .subtitle {
          font-size: 1rem;
          color: #666;
          margin-bottom: 1.5rem;
        }

        textarea {
          width: 100%;
          max-width: 500px;
          padding: 0.75rem;
          font-size: 1rem;
          border: 2px solid #4e54c8;
          border-radius: 8px;
          outline: none;
          resize: none;
          overflow: hidden;
          transition: height 0.3s ease, box-shadow 0.3s ease;
          min-height: 80px;
        }

        textarea:focus {
          box-shadow: 0 0 8px rgba(78, 84, 200, 0.2);
        }

        button {
          margin-top: 1.5rem;
          padding: 0.75rem 1.5rem;
          font-size: 1rem;
          font-weight: bold;
          color: #fff;
          background: linear-gradient(135deg, #4e54c8, #8f94fb);
          border: none;
          border-radius: 8px;
          cursor: pointer;
          box-shadow: 0 5px 15px rgba(78, 84, 200, 0.2);
          transition: background 0.3s ease, transform 0.3s ease;
        }

        button:hover {
          background: linear-gradient(135deg, #8f94fb, #4e54c8);
          transform: translateY(-2px);
        }

        button:active {
          transform: translateY(1px);
        }

        p#result {
          margin-top: 1.5rem;
          font-size: 1.25rem;
          font-weight: bold;
          color: #4e54c8;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>📰 Fake News Detector</h1>
            <p class="subtitle">Check if a piece of news is fake or real with a quick analysis!</p>
            <textarea id="text" rows="1" placeholder="Paste the news text here..." oninput="autoResizeTextarea(this)"></textarea>
            <button onclick="getPrediction()">Analyze News</button>
            <p id="result"></p>
        </div>
    </div>
</body>
</html>

"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('text')
    if not data:
        return jsonify({'error': 'No input text provided'}), 400

    l=[]
    inputData=np.array([data])
    for idx, model in enumerate(models):
        # print(model,data)
        try:
            prediction = model.predict(inputData)
            l.append(prediction[0])
            # print(f"Prediction from model {idx + 1} ({type(model).__name__}): {prediction}")
        except AttributeError as e:
            print(f"Error: Model {idx + 1} ({type(model).__name__}) might not have a 'predict' method or has input shape mismatch.")
            print(e)
    # print(l)
    if l.count(1)>3:
        # print("True")
        return jsonify({'result': "True"})
    else:
        # print("False")
        return jsonify({'result': "False"})

    
if __name__ == '__main__':
    app.run(debug=True)
