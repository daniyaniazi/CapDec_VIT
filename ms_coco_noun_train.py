import clip
import torch
import json
import pickle

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-L/14", device=device)

# Load Custom Words from JSON
with open("dataset/words.json", "r") as f:
    words = json.load(f)

# Convert Words into CLIP Text Embeddings
text_tokens = clip.tokenize(words).to(device)
text_embeddings = clip_model.encode_text(text_tokens).cpu()

# Save as a Pickle File for CapDec
data = {
    "clip_embedding_text_dave": text_embeddings,
    "captions": [{"caption": word} for word in words]
}

with open("custom_clip_embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Saved CLIP embeddings in 'custom_clip_embeddings.pkl'")
