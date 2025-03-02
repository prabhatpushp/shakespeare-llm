import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Hyperparameters for the model
BATCH_SIZE = 32          # Number of sequences to process in parallel
BLOCK_SIZE = 256        # Maximum context length for predictions
MAX_EPOCHS = 3000       # Number of training iterations
LEARNING_RATE = 3e-4    # How quickly the model learns
EVAL_INTERVAL = 500     # How often to evaluate the model
EVAL_ITERS = 200        # Number of iterations for evaluation
N_EMBED = 384          # Size of embedding vectors
N_HEAD = 6             # Number of attention heads
N_LAYER = 6            # Number of transformer layers
DROPOUT = 0.2          # Dropout rate for regularization

# Set random seed for reproducibility
torch.manual_seed(1337)

# Check if GPU is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Read and preprocess the Shakespeare text
def read_shakespeare():
    with open('shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Get all unique characters from the text and create encoding/decoding mappings
class TextEncoder:
    def __init__(self, text):
        # Get unique characters from text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        # Create character to integer mapping
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        # Create integer to character mapping
        self.itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, string):
        # Convert string to list of integers
        return [self.stoi[c] for c in string]
    
    def decode(self, integers):
        # Convert list of integers back to string
        return ''.join([self.itos[i] for i in integers])

# Define the transformer model
class ShakespeareModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Token embedding table
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        # Position embedding table
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        # Transformer layers
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYER)])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(N_EMBED)
        # Final linear layer to predict next token
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Get token embeddings
        tok_emb = self.token_embedding(idx)
        # Get position embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        # Combine token and position embeddings
        x = tok_emb + pos_emb
        # Apply transformer blocks
        x = self.blocks(x)
        # Apply final layer norm
        x = self.ln_f(x)
        # Get logits through final linear layer
        logits = self.lm_head(x)
        
        # If targets are provided, compute loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # Generate new tokens one at a time
        for _ in range(max_new_tokens):
            # Crop context to block size
            idx_cond = idx[:, -BLOCK_SIZE:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus on last time step
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Define a single transformer block
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Multi-head attention
        self.sa = MultiHeadAttention()
        # Feed-forward network
        self.ffwd = FeedForward()
        # Layer normalizations
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)
    
    def forward(self, x):
        # Apply attention with residual connection
        x = x + self.sa(self.ln1(x))
        # Apply feed-forward with residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

# Define multi-head attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Define key, query, value projections
        self.c_attn = nn.Linear(N_EMBED, 3 * N_EMBED)
        # Define output projection
        self.c_proj = nn.Linear(N_EMBED, N_EMBED)
        # Define dropout
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        # Create attention mask
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        
    def forward(self, x):
        B, T, C = x.shape
        # Get query, key, value vectors
        q, k, v = self.c_attn(x).split(N_EMBED, dim=2)
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(C)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # Apply attention to values
        y = att @ v
        # Apply output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

# Define feed-forward network
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # Define feed-forward layers
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )
    
    def forward(self, x):
        return self.net(x)

# Training function
def train_model():
    # Read the text data
    text = read_shakespeare()
    print(f"Text length: {len(text)}")
    
    # Create the encoder
    encoder = TextEncoder(text)
    print(f"Vocabulary size: {encoder.vocab_size}")
    
    # Encode the entire text
    data = torch.tensor(encoder.encode(text), dtype=torch.long)
    
    # Split into train and validation sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Create the model
    model = ShakespeareModel(encoder.vocab_size)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for iter in range(MAX_EPOCHS):
        # Sample a batch of data
        xb, yb = get_batch(train_data)
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Print progress
        if iter % EVAL_INTERVAL == 0:
            print(f"Step {iter}: train loss {loss.item():.4f}")
            
            # Generate some text
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(encoder.decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# Function to get a batch of data
def get_batch(data):
    # Generate random starting indices
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    # Get input sequences
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    # Get target sequences (shifted by 1)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    # Move to device
    x, y = x.to(device), y.to(device)
    return x, y

if __name__ == '__main__':
    train_model()
