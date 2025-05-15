import torch
from sklearn.model_selection import train_test_split

# Load embeddings
eeg_embeddings = torch.load("eeg_embeddings.pt")  # Tensor: [N, 256]
text_data = torch.load("text_embeddings.pt")      # Dict with 'embeddings' + 'words'

text_embeddings = text_data['embeddings']         # Tensor: [N, 768]
text_words = text_data['words']                   # List of N strings

# === Train-Test Split (on aligned triplets) ===
eeg_train, eeg_test, emb_train, emb_test, words_train, words_test = train_test_split(
    eeg_embeddings, text_embeddings, text_words,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# === Save all ===
torch.save({'eeg': eeg_train, 'text': emb_train, 'words': words_train}, 'train_embeddings.pt')
torch.save({'eeg': eeg_test, 'text': emb_test, 'words': words_test}, 'test_embeddings.pt')

print("Saved train and test splits with raw text!")
