"""MuST-Guard teacher (PROPOSAL.md 3.2, PROTOCOL.md section 3 and A8): four modality encoders, gated fusion and a
spam head. Conformal abstention is applied afterwards to the saved scores (score_mg.py).

- body: a pretrained transformer encoder, first-token state.
- header and URL strings: byte-level char-CNNs; the URL branch also takes 11 lexical features.
- struct: an MLP on 6 HTML structure counts.
Fusion: each branch is projected to d; a scalar gate per branch is computed from its own vector, gates of missing
branches are masked, and the softmax-weighted sum feeds the head. The body branch is never missing, so the model
runs on corpora without headers or URLs. The gate weights are returned for the branch analysis.
A11 T3: an auxiliary head on the body vector alone; its logits are kept in aux_logits after each forward pass.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

N_BYTES = 257  # 256 byte values + padding 0 (bytes are stored +1)


def encode_bytes(texts, max_chars):
    x = torch.zeros(len(texts), max_chars, dtype=torch.long)
    for i, t in enumerate(texts):
        b = t.encode("utf-8", "ignore")[:max_chars]
        if b:
            x[i, :len(b)] = torch.tensor(list(b)) + 1
    return x


class CharCNN(nn.Module):
    def __init__(self, d, emb=32, ch=64, widths=(3, 5, 7)):
        super().__init__()
        self.emb = nn.Embedding(N_BYTES, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, ch, w, padding=w // 2) for w in widths)
        self.out = nn.Linear(ch * len(widths), d)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).amax(dim=2) for c in self.convs], dim=1)
        return self.out(h)


def mlp(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))


class MuSTGuard(nn.Module):
    BRANCHES = ("body", "header", "url", "struct")

    def __init__(self, encoder, d=256, n_lex=11, n_struct=6, use=("body", "header", "url", "struct"), dtype=None):
        super().__init__()
        self.use = set(use)
        self.enc = AutoModel.from_pretrained(encoder, dtype=dtype)
        self.p_body = nn.Linear(self.enc.config.hidden_size, d)
        self.hdr = CharCNN(d)
        self.url = CharCNN(d)
        self.url_lex = mlp(n_lex, d)
        self.p_url = nn.Linear(2 * d, d)
        self.struct = mlp(n_struct, d)
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        self.aux_head = nn.Linear(d, 2)  # body-only head of A11 T3; unused when its loss weight is 0

    def forward(self, enc, hdr, url, lex, struct, mask):
        """enc: tokenizer output; hdr, url: byte tensors; lex, struct: float tensors;
        mask: (B, 4) float, 1 = branch present and used."""
        h_body = self.p_body(self.enc(**enc).last_hidden_state[:, 0])
        hs = [h_body, self.hdr(hdr), self.p_url(torch.cat([self.url(url), self.url_lex(lex)], 1)),
              self.struct(struct)]
        h = torch.stack([x.float() for x in hs], dim=1)  # (B, 4, d)
        use = torch.tensor([b in self.use for b in self.BRANCHES], device=h.device, dtype=mask.dtype)
        m = mask * use
        s = self.gate(h).squeeze(-1).masked_fill(m == 0, float("-inf"))
        a = torch.softmax(s, dim=1)
        z = self.norm((a.unsqueeze(-1) * h).sum(1))
        self.aux_logits = self.aux_head(h_body.float())  # read by train_mg.py for the T3 loss
        return self.head(z), a
