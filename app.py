from flask import Flask, render_template, request, jsonify
import torch
import sentencepiece as spm
from transformer import Transformer  # Import Transformer class from transformer.py


# Initialize Flask app
app = Flask(__name__)



# Load SentencePiece tokenizer
tokenizer = spm.SentencePieceProcessor()
try:
    tokenizer.load("urdu_english_bpe_parallel.model")
except Exception as e:
    print(f"Error loading tokenizer model: {e}")




# Define constants
PAD_IDX = tokenizer.piece_to_id("<pad>")
MAX_SEQ_LEN = 20





# Define model parameters
src_vocab_size = len(tokenizer)
tgt_vocab_size = len(tokenizer)
src_pad_idx = PAD_IDX
tgt_pad_idx = PAD_IDX
d_model = 128
num_heads = 2
num_layers = 2
d_ff = 512
max_len = 200
dropout = 0.1







# Initialize Transformer model
device = torch.device("cpu")
model = Transformer(
    src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx,
    d_model, num_heads, num_layers, d_ff, max_len
)







try:
    model.load_state_dict(torch.load("transformer_translation1.pth", map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model weights: {e}")







# Helper function to tokenize and pad input sentence
def preprocess_sentence(sentence):
    tokenized = tokenizer.encode(sentence, out_type=int)
    tokenized = tokenized[:MAX_SEQ_LEN] + [PAD_IDX] * (MAX_SEQ_LEN - len(tokenized))
    return torch.tensor([tokenized], dtype=torch.long)









# Function to generate translation
def translate(sentence):
    src = preprocess_sentence(sentence).to(device)
    tgt_input = torch.tensor([[PAD_IDX]], dtype=torch.long).to(device)

    for _ in range(MAX_SEQ_LEN):
        output = model(src, tgt_input)
        next_token = output.argmax(dim=-1)[:, -1].item()
        if next_token == PAD_IDX:
            break
        tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)

    # Decode token IDs to Urdu text
    translation = tokenizer.decode(tgt_input.squeeze().tolist()[1:])
    return translation

# Translation history
history = []








@app.route("/", methods=["GET", "POST"])
def index():
    global history
    if request.method == "POST":
        data = request.get_json()
        english_text = data.get("english_text", "").strip()
        if not english_text:
            return jsonify({"error": "No input provided"}), 400

        try:
            urdu_translation = translate(english_text)
            history.append({"source": english_text, "target": urdu_translation})
            return jsonify({"translation": urdu_translation, "history": history})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("index.html", history=history)












if __name__ == "__main__":
    app.run(debug=True)
