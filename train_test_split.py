import torch
from sklearn.model_selection import train_test_split

# Load your existing embeddings
eeg_embeddings = torch.load("eeg_embeddings.pt")
text_embeddings = torch.load("text_embeddings.pt")
text_embeddings = text_embeddings['embeddings']  # if it's a dict

# Train-Test Split
eeg_train, eeg_test, text_train, text_test = train_test_split(
    eeg_embeddings, text_embeddings, 
    test_size=0.2,  # 80% train, 20% test
    random_state=42,
    shuffle=True
)

# Save splits
torch.save({'eeg': eeg_train, 'text': text_train}, 'train_embeddings.pt')
torch.save({'eeg': eeg_test, 'text': text_test}, 'test_embeddings.pt')

print("Saved train and test splits!")
