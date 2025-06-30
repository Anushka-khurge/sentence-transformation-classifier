# sentence-transformation-classifier
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sentence Transformation Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 40px; line-height: 1.6; background: #f8f9fa; color: #333; }
    h1, h2, h3 { color: #1a73e8; }
    code { background: #eee; padding: 2px 4px; border-radius: 3px; }
    pre { background: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto; }
    ul { margin-bottom: 1rem; }
  </style>
</head>
<body>

<h1>🧠 Sentence Transformation Classifier using BERT</h1>

<p>This project uses a fine-tuned BERT model to classify English sentences into six types of transformations:</p>
<ul>
  <li>Active to Passive</li>
  <li>Passive to Active</li>
  <li>Direct to Indirect</li>
  <li>Indirect to Direct</li>
  <li>Positive to Negative</li>
  <li>Negative to Positive</li>
</ul>

<p>The model is trained on a custom dataset of 1000 sentence pairs and integrated into an interactive Streamlit web app with attention-based explainability.</p>

<hr>

<h2>🔍 Objective</h2>
<p>To classify transformed sentences into one of six grammatical transformation types using a fine-tuned BERT model and visualize the model’s reasoning using attention maps.</p>

<h2>🧠 Model & Approach</h2>
<ul>
  <li><strong>Model:</strong> BERT (bert-base-uncased)</li>
  <li><strong>Library:</strong> Hugging Face Transformers & PyTorch</li>
  <li><strong>Input:</strong> Transformed Sentence</li>
  <li><strong>Output:</strong> Predicted Transformation Label</li>
  <li><strong>Explainability:</strong> Average attention visualization from CLS token</li>
</ul>

<h2>🗂️ Project Structure</h2>
<pre>
📁 finetuned_model/           # Saved fine-tuned model
📁 data/                      # CSV of original + transformed sentences
📁 logs/, 📁 results/          # Optional training logs
📄 app.py                     # Streamlit web app
📄 train_and_evaluate.ipynb   # Notebook for training, evaluation, explainability
📄 README.html                # Project documentation
</pre>

<h2>⚙️ Installation</h2>
<pre><code>
# Clone the repo
git clone https://github.com/your-username/sentence-transformation-classifier.git
cd sentence-transformation-classifier

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\activate on Windows

# Install required libraries
pip install -r requirements.txt
</code></pre>

<h2>📦 Required Libraries (Important)</h2>
<pre><code>
transformers
torch
scikit-learn
streamlit
seaborn
matplotlib
pandas
numpy
</code></pre>

<h2>🚀 Run the Streamlit App</h2>
<pre><code>streamlit run app.py</code></pre>

<h2>📊 Model Evaluation</h2>
<ul>
  <li>Precision, Recall, F1-Score per transformation class</li>
  <li>Classification Report + Confusion Matrix</li>
  <li>Attention heatmaps for explainability</li>
</ul>

<h2>✅ Sample Output</h2>
<ul>
  <li><strong>Input Sentence:</strong> They do not like coffee.</li>
  <li><strong>Predicted Transformation:</strong> Negative to Positive</li>
  <li><strong>Confidence:</strong> 93.5%</li>
  <li><strong>Attention:</strong> Highlights important words like “not”, “like”</li>
</ul>

<h2>📋 Task Completion Checklist</h2>
<ul>
  <li>✅ Custom dataset created (1000 rows)</li>
  <li>✅ BERT fine-tuned on transformation labels</li>
  <li>✅ Evaluation metrics + confusion matrix</li>
  <li>✅ Attention-based explainability</li>
  <li>✅ Streamlit app with prediction + visualization</li>
</ul>

<h2>🙋‍♀️ Author</h2>
<p><strong>Anushka Khurge</strong><br>
B.Tech in Artificial Intelligence<br>
</p>


</body>
</html>
