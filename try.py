import torch
import re
import torch.nn as nn
import difflib
import streamlit as st

# Define the path to the text file containing Paul Graham's essays
filepath = 'paul_graham_essays.txt'

# Read and preprocess the text
with open(filepath, 'r') as file:
    essay_text = file.read()
essay_text = re.sub(r'\.+', ' . ', essay_text)
essay_text = re.sub(r'\d+', '', essay_text)
essay_text = re.sub('[^a-zA-Z \.]', '', essay_text).lower()
words = essay_text.split()

# Create vocabulary mappings
unique_words = sorted(set(words) - {'.'})
stoi = {w: i + 1 for i, w in enumerate(unique_words)}
stoi['.'] = 0
itos = {i: w for w, i in stoi.items()}
itos[0] = '.'

# Define model classes
class NextWordMLP(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation_choice):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        self.activation = nn.ReLU() if activation_choice == 'ReLu' else nn.Tanh()

    def forward(self, x):
        x = self.emb(x).view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        return self.lin2(x)

# Handle out-of-vocabulary (OOV) words
def replace_oov_words(input_text):
    processed_words = []
    for word in input_text.split():
        if word in stoi:
            processed_words.append(word)
        else:
            similar_words = difflib.get_close_matches(word, list(stoi.keys()), n=1)
            replacement = similar_words[0] if similar_words else '<OOV>'
            processed_words.append(replacement)
    return processed_words

# Predict next words function
def predict_next_words(input_text, k, model, block_size):
    processed_words = replace_oov_words(input_text)
    input_indices = [stoi[word] for word in processed_words[-block_size:]]
    input_tensor = torch.tensor(input_indices).unsqueeze(0)

    with torch.no_grad():
        predictions = model(input_tensor)
    top_k_indices = torch.topk(predictions[0, :], k).indices
    return [itos[idx.item()] for idx in top_k_indices]

# Streamlit setup
st.set_page_config(page_title="Next Word Predictor", layout="centered")
st.markdown("## 🌸Next k word prediction app🌸")
st.sidebar.header("🔧 Settings")

embedding_size = st.sidebar.selectbox("Embedding Size", ["32", "64", "128"])
activation_choice = st.sidebar.selectbox("Select Activation Function", ["ReLu", "Tanh"])
context_options = ["3", "5"] if activation_choice == "ReLu" else ["5", "10"]
context_length = st.sidebar.selectbox("Context Length", context_options)

block_size = int(context_length)
input_text = st.sidebar.text_input("Input text", "")
num_words_to_predict = st.sidebar.text_input("Number of Words to predict", "")

# Prediction button
if st.button("Predict"):
    try:
        # Initialize and load model based on user input
        model = NextWordMLP(block_size, len(stoi), int(embedding_size), 10, activation_choice)
        model_file = f"model_e{embedding_size}_c{context_length}_{activation_choice}.pt"
        model.load_state_dict(torch.load(model_file), strict=False)
        model.eval()

        # Run prediction
        predictions = predict_next_words(input_text, int(num_words_to_predict), model, block_size)
        st.write(predictions)
    except Exception as e:
        st.error(f"An error occurred: {e}")
