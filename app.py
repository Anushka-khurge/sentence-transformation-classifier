import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Label classes (same order as training)
labels = [
    "Active to Passive",
    "Passive to Active",
    "Direct to Indirect",
    "Indirect to Direct",
    "Positive to Negative",
    "Negative to Positive"
]

# Load fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("finetuned_model/", output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained("finetuned_model/")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("🧠 Sentence Transformation Classifier with Explainability")
st.markdown("Enter a **transformed sentence**, and I’ll predict the type of transformation applied along with an attention plot!")

# Text input
input_text = st.text_area("Enter transformed sentence:")

# On click
if st.button("Predict"):
    if not input_text.strip():
        st.warning("Please enter a sentence.")
    else:
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        # Get prediction + attention
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = labels[pred_idx]
            confidence = probs[0][pred_idx].item() * 100
            attentions = outputs.attentions

        # ✅ Output prediction
        st.success(f"**Predicted Transformation:** {pred_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # ✅ Process attention (CLS token to each word, last 4 layers avg)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        cls_attention = torch.stack(attentions[-4:]).mean(dim=0)[0, :, 0, :].mean(dim=0).numpy()

        # ✅ Plot attention
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.barplot(x=tokens, y=cls_attention, ax=ax)
        ax.set_title("Avg Attention from [CLS] to Each Token (Last 4 Layers, All Heads)")
        ax.set_ylabel("Attention Score")
        ax.set_xlabel("Tokens")
        plt.xticks(rotation=45)
        st.pyplot(fig)
