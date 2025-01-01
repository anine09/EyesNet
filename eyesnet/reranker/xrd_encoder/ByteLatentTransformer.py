import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Custom Layer Normalization."""

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    """Transformer Block with LayerNorm, Attention, and MLP."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd, num_heads=n_head, dropout=dropout, batch_first=True
        )
        self.ln2 = LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttentionLayer(nn.Module):
    """Cross Attention Layer for Encoder and Decoder."""

    def __init__(self, query_dim, key_dim, n_head, dropout):
        super().__init__()
        self.ln_q = LayerNorm(query_dim)
        self.ln_kv = LayerNorm(key_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=n_head, dropout=dropout, batch_first=True
        )
        self.proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        query = self.ln_q(query)
        key = self.ln_kv(key)
        value = self.ln_kv(value)
        attn_out, _ = self.attn(query, key, value)
        attn_out = self.proj(attn_out)
        attn_out = self.dropout(attn_out)
        return query + attn_out


class HashNGramEmbedding(nn.Module):
    """Hash n-gram Embeddings."""

    def __init__(self, hash_sizes, hash_table_size, n_embd):
        super().__init__()
        self.hash_sizes = hash_sizes
        self.hash_table_size = hash_table_size
        self.n_embd = n_embd
        self.hash_embeddings = nn.ModuleDict(
            {f"hash_{n}": nn.Embedding(hash_table_size, n_embd) for n in hash_sizes}
        )

    def forward(self, x):
        B, T = x.shape
        embeddings = torch.zeros(B, T, self.n_embd, device=x.device)
        for n in self.hash_sizes:
            if T < n:
                continue
            # Extract n-grams
            ngrams = x.unfold(1, n, 1)  # [B, T - n +1, n]
            # Compute hash
            hashes = self.roll_poly_hash(ngrams)
            hashes = hashes % self.hash_table_size
            # Lookup embeddings
            hash_emb = self.hash_embeddings[f"hash_{n}"](
                hashes
            )  # [B, T - n +1, n_embd]
            # Scatter add
            embeddings[:, n - 1 : T, :] += hash_emb
        # Normalize
        embeddings = embeddings / len(self.hash_sizes)
        return embeddings  # [B, T, n_embd]

    def roll_poly_hash(self, ngrams):
        """Simple polynomial rolling hash."""
        base = 257
        hash_val = torch.zeros(
            ngrams.size(0), ngrams.size(1), device=ngrams.device, dtype=torch.long
        )
        for i in range(ngrams.size(2)):
            hash_val = (hash_val * base + ngrams[:, :, i].long()) % (2**32)
        return hash_val


class LocalEncoder(nn.Module):
    """Local Encoder that encodes input bytes into patch representations."""

    def __init__(
        self,
        vocab_size,
        n_embd,
        patch_size,
        hash_sizes,
        hash_table_size,
        n_head,
        dropout,
        lE,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_embd = n_embd
        self.byte_embedding = nn.Embedding(vocab_size, n_embd)
        self.hash_ngram = HashNGramEmbedding(hash_sizes, hash_table_size, n_embd)
        self.transformer_blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(lE)]
        )
        self.cross_attn = CrossAttentionLayer(n_embd, n_embd, n_head, dropout)
        self.ln = LayerNorm(n_embd)

    def forward(self, x):
        B, T = x.shape
        # Byte Embedding
        x_emb = self.byte_embedding(x)  # [B, T, C]
        # Hash n-gram Embedding
        hash_emb = self.hash_ngram(x)  # [B, T, C]
        x_emb = x_emb + hash_emb  # [B, T, C]
        # Transformer Layers
        for block in self.transformer_blocks:
            x_emb = block(x_emb)
        # Cross-Attention to form patches
        # Assume patches are non-overlapping
        # Pad if necessary
        if T % self.patch_size != 0:
            pad_len = self.patch_size - (T % self.patch_size)
            pad = torch.zeros((B, pad_len), dtype=x.dtype, device=x.device).long()
            pad_emb = self.byte_embedding(pad)  # [B, pad_len, C]
            pad_emb += self.hash_ngram(pad)  # Incorporate hash embeddings
            x_emb = torch.cat([x_emb, pad_emb], dim=1)  # [B, T + pad_len, C]
            T += pad_len
        # Reshape and pool to create patch representations
        patches = x_emb.view(
            B, T // self.patch_size, self.patch_size, self.n_embd
        ).mean(
            dim=2
        )  # [B, N_patches, C]
        patches = self.cross_attn(patches, x_emb, x_emb)  # [B, N_patches, C]
        patches = self.ln(patches)
        return patches  # [B, N_patches, C]


class LatentTransformer(nn.Module):
    """Latent Transformer over patch representations."""

    def __init__(self, n_embd, n_head, n_layers, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(n_embd)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


class ByteLatentTransformer(nn.Module):
    """Byte Latent Transformer combining encoder, transformer, and decoder."""

    def __init__(
        self,
        vocab_size,
        n_embd,
        n_head,
        n_layers_encoder,
        n_layers_latent,
        dropout,
        patch_size,
        hash_sizes,
        hash_table_size,
        block_size,
    ):
        super().__init__()
        self.local_encoder = LocalEncoder(
            vocab_size,
            n_embd,
            patch_size,
            hash_sizes,
            hash_table_size,
            n_head,
            dropout,
            n_layers_encoder,
        )
        self.latent_transformer = LatentTransformer(
            n_embd, n_head, n_layers_latent, dropout
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, block_size // patch_size, n_embd)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Encode bytes to patches
        patches = self.local_encoder(x)  # [B, N_patches, C]
        # Add positional embeddings
        patches = (
            patches + self.pos_embedding[:, : patches.size(1), :]
        )  # [B, N_patches, C]
        # Transform patches
        transformed_patches = self.latent_transformer(patches)  # [B, N_patches, C]

        return transformed_patches
