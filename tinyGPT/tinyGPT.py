import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64
block_size = 64
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
device = 'cpu'
eval_iters = 200
train_split_size = 0.9
n_embd = 64
n_decoder_blocks = 6
dropout = 0.2
##################


torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Build the vocab from the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenizer
# Create a mapping from char to ints
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


# Encode the whole dataset and create training / validation splits
data = torch.tensor(encode(text), dtype=torch.long)
split = int(train_split_size * len(text))
train_data = data[:split]
val_data = data[split:]


def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])

    # Move to device
    x = x.to(device)
    y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size

        # Create the query, key and value linear layers
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)

        # A state for the module to remember the lower triangular matrix, that's not considered a parameter.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.k(x)  # (B, T, head_size)
        q = self.q(x)  # (B, T, head_size)
        v = self.v(x)  # (B, T, head_size)

        # Compute the dot product of the query and key
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # Compute the weighted sum of the values
        out = wei @ v  # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([eachHead(x) for eachHead in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout_layer(out)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()

        # Communication layer
        self.attention_heads = MultiHeadAttention(n_heads, n_embd//n_heads)

        # Linear Layer for Computation
        self.ffll = FeedForward(n_embd)

        # Layer Normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Attention Layer
        x = self.attention_heads(self.ln1(x)) + x  # Plus x is the residual connection

        # Linear Layer
        x = self.ffll(self.ln2(x)) + x # Plus x is the residual connection

        return x

    


# Create a simple Bigram Language Model
class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)     # Identity of the token
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # Position of the token

        n_heads = 4

        # Attention and Computation Block
        self.block = nn.Sequential(*[DecoderBlock(n_embd, n_heads) for _ in range(n_decoder_blocks)])

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(n_embd)

        # Linear Layer for the output
        self.ln_head = nn.Linear(n_embd, vocab_size)                   

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Get the token embeddings from the embedding table
        token_embeddings = self.token_embedding_table(idx)       # (B, T, C / embedding_size)

        # Get the position embeddings from the embedding table (Sample out their positions)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T, C / embedding_size)

        # The sum of the token and position embeddings is the input to the transformer
        x = token_embeddings + position_embeddings

        # Pass it through the attention head
        x = self.block(x)                  # (B, T, C / embedding_size)

        # Pass it through the layer normalization
        x = self.layer_norm(x)             # (B, T, C / embedding_size)

        # Get the logits from the linear layer
        logits = self.ln_head(x)                  # (B, T, vocab_size)

        B, T, C = logits.shape

        # Re-arrange the logits to be of shape (batch_size * block_size, vocab_size)
        logits = logits.view(B*T, C)

        if targets is None:
            loss = None

        else:
            targets = targets.view(B*T)

            # Compute the loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]             # (B, T)

            # Get the predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[-1, :].view(1, -1)

            # Apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)                      # (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)     # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)                # (B, T + 1)

        return idx


model = TransformerLM().to(device)

# Print the number of parameters in the model
print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

# Create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
for iter in range(max_iters):
    # Every once in a while, evaluate the model
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f'Iter {iter} | Train Loss {losses["train"]:.4f} | Val Loss {losses["val"]:.4f}')

    # Sample a batch of data
    X, Y = get_batch('train')

    # Get the logits and loss
    logits, loss = model(X, Y)

    # Backprop
    optimizer.zero_grad(set_to_none=True)

    # Populate gradients for all trainable parameters
    loss.backward()

    # Update the parameters
    optimizer.step()


# Generate some text
starting_point = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate 1000 tokens
following_chars = model.generate(starting_point, 1000)

# Decode the tokens
print(decode(following_chars[0].tolist()))
