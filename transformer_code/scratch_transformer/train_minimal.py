import argparse
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import corpus_bleu

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


@dataclass
class Config:
    data_dir: str = str(ROOT / "data" / "multi30k-de-en")
    save_dir: str = str(ROOT / "checkpoints" / "scratch_transformer")
    epochs: int = 20
    batch_size: int = 64
    max_len: int = 80
    max_vocab: int = 10000
    min_freq: int = 2
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    warmup_steps: int = 2000
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    bleu_samples: int = 0
    seed: int = 42
    attention_mode: str = "qkv"  # qkv | shared_kv


def tokenize(text):
    return text.lower().strip().split()


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


class Vocab:
    def __init__(self, texts, max_size, min_freq):
        counter = Counter(tok for line in texts for tok in tokenize(line))
        words = [w for w, c in counter.most_common() if c >= min_freq]
        self.itos = [PAD, BOS, EOS, UNK] + words[: max_size - 4]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_idx = self.stoi[PAD]
        self.bos_idx = self.stoi[BOS]
        self.eos_idx = self.stoi[EOS]
        self.unk_idx = self.stoi[UNK]

    def encode(self, text):
        return [self.bos_idx] + [self.stoi.get(tok, self.unk_idx) for tok in tokenize(text)] + [self.eos_idx]

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.itos[int(i)]
            if w == EOS:
                break
            if w not in (PAD, BOS):
                words.append(w)
        return words


class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, src_vocab, tgt_vocab, max_len):
        self.pairs = []
        for src, tgt in zip(src_lines, tgt_lines):
            s = src_vocab.encode(src)[:max_len]
            t = tgt_vocab.encode(tgt)[:max_len]
            if len(s) > 2 and len(t) > 2:
                self.pairs.append((s, t))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch, pad_idx):
    src, tgt = zip(*batch)
    src_len = max(len(x) for x in src)
    tgt_len = max(len(x) for x in tgt)
    src_tensor = torch.full((len(batch), src_len), pad_idx, dtype=torch.long)
    tgt_tensor = torch.full((len(batch), tgt_len), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src_tensor[i, : len(s)] = torch.tensor(s)
        tgt_tensor[i, : len(t)] = torch.tensor(t)
    return src_tensor, tgt_tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        q = self.split(self.q(q))
        k = self.split(self.k(k))
        v = self.split(self.v(v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(q.size(0), -1, self.n_heads * self.d_k)
        return self.o(x)


class MultiHeadAttentionSharedKV(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        q = self.split(self.q(q))
        kv = self.split(self.kv(k))
        scores = q @ kv.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        x = attn @ kv
        x = x.transpose(1, 2).contiguous().view(q.size(0), -1, self.n_heads * self.d_k)
        return self.o(x)


def build_attention(cfg):
    if cfg.attention_mode == "shared_kv":
        return MultiHeadAttentionSharedKV(cfg.d_model, cfg.n_heads, cfg.dropout)
    if cfg.attention_mode == "qkv":
        return MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
    raise ValueError("attention_mode must be 'qkv' or 'shared_kv'")


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg, decoder=False):
        super().__init__()
        self.decoder = decoder
        self.self_attn = build_attention(cfg)
        self.cross_attn = build_attention(cfg) if decoder else None
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.n3 = nn.LayerNorm(cfg.d_model) if decoder else None
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, src_mask=None, memory=None, tgt_mask=None):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, tgt_mask if self.decoder else src_mask)))
        if self.decoder:
            x = self.n2(x + self.drop(self.cross_attn(x, memory, memory, src_mask)))
            x = self.n3(x + self.drop(self.ff(x)))
            return x
        x = self.n2(x + self.drop(self.ff(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg, src_vocab_size, tgt_vocab_size, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_emb = nn.Embedding(src_vocab_size, cfg.d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, cfg.d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.enc = nn.ModuleList([Block(cfg, decoder=False) for _ in range(cfg.n_layers)])
        self.dec = nn.ModuleList([Block(cfg, decoder=True) for _ in range(cfg.n_layers)])
        self.out = nn.Linear(cfg.d_model, tgt_vocab_size)
        self.scale = math.sqrt(cfg.d_model)

    def src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def tgt_mask(self, tgt):
        pad = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        n = tgt.size(1)
        causal = torch.tril(torch.ones(n, n, device=tgt.device)).bool()
        return pad & causal.unsqueeze(0).unsqueeze(0)

    def encode(self, src):
        m = self.src_mask(src)
        x = self.drop(self.pos(self.src_emb(src) * self.scale))
        for layer in self.enc:
            x = layer(x, src_mask=m)
        return x, m

    def decode(self, tgt, memory, src_mask):
        tm = self.tgt_mask(tgt)
        x = self.drop(self.pos(self.tgt_emb(tgt) * self.scale))
        for layer in self.dec:
            x = layer(x, src_mask=src_mask, memory=memory, tgt_mask=tm)
        return x

    def forward(self, src, tgt):
        memory, sm = self.encode(src)
        return self.out(self.decode(tgt, memory, sm))


def noam_lr(step, d_model, warmup):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(cfg):
    data = Path(cfg.data_dir)
    tr_en, tr_de = read_lines(data / "train.en"), read_lines(data / "train.de")
    va_en, va_de = read_lines(data / "val.en"), read_lines(data / "val.de")
    te_en, te_de = read_lines(data / "test.en"), read_lines(data / "test.de")
    src_vocab = Vocab(tr_en, cfg.max_vocab, cfg.min_freq)
    tgt_vocab = Vocab(tr_de, cfg.max_vocab, cfg.min_freq)
    train = TranslationDataset(tr_en, tr_de, src_vocab, tgt_vocab, cfg.max_len)
    val = TranslationDataset(va_en, va_de, src_vocab, tgt_vocab, cfg.max_len)
    test = TranslationDataset(te_en, te_de, src_vocab, tgt_vocab, cfg.max_len)
    if cfg.train_samples:
        train.pairs = train.pairs[: cfg.train_samples]
    if cfg.val_samples:
        val.pairs = val.pairs[: cfg.val_samples]
    if cfg.test_samples:
        test.pairs = test.pairs[: cfg.test_samples]
    return train, val, test, src_vocab, tgt_vocab


def greedy_decode(model, src, src_vocab, tgt_vocab, max_len, device):
    model.eval()
    src = src.unsqueeze(0).to(device)
    memory, sm = model.encode(src)
    ys = torch.tensor([[tgt_vocab.bos_idx]], device=device)
    for _ in range(max_len - 1):
        logits = model.out(model.decode(ys, memory, sm))[:, -1]
        nxt = logits.argmax(-1).item()
        ys = torch.cat([ys, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == tgt_vocab.eos_idx:
            break
    return tgt_vocab.decode(ys[0].tolist())


@torch.no_grad()
def evaluate(model, loader, criterion, tgt_vocab, device, bleu=False, src_vocab=None, max_len=80, bleu_samples=0):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hyps, refs = [], []
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tin, ty = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tin)
        loss = criterion(logits.reshape(-1, logits.size(-1)), ty.reshape(-1))
        tok = (ty != tgt_vocab.pad_idx).sum().item()
        total_loss += loss.item() * tok
        total_tokens += tok
        if bleu:
            for s, r in zip(src.cpu(), tgt.cpu()):
                if bleu_samples and len(hyps) >= bleu_samples:
                    break
                hyps.append(greedy_decode(model, s, src_vocab, tgt_vocab, max_len, device))
                refs.append(tgt_vocab.decode(r.tolist()[1:]))
    ce = total_loss / max(total_tokens, 1)
    ppl = math.exp(ce)
    return ce, ppl, corpus_bleu(hyps, refs) if bleu else 0.0


def train(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set, src_vocab, tgt_vocab = load_data(cfg)
    collate_fn = lambda b: collate(b, src_vocab.pad_idx)
    train_loader = DataLoader(train_set, cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Transformer(cfg, len(src_vocab.itos), len(tgt_vocab.itos), src_vocab.pad_idx).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx, reduction="mean")
    optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} attention_mode={cfg.attention_mode}")
    print(f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    best_bleu = -1.0
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tin, ty = tgt[:, :-1], tgt[:, 1:]
            step += 1
            lr = noam_lr(step, cfg.d_model, cfg.warmup_steps)
            for g in optim.param_groups:
                g["lr"] = lr
            logits = model(src, tin)
            loss = criterion(logits.reshape(-1, logits.size(-1)), ty.reshape(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tok = (ty != tgt_vocab.pad_idx).sum().item()
            total_loss += loss.item() * tok
            total_tokens += tok

        train_ce = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(train_ce)
        val_ce, val_ppl, val_bleu = evaluate(
            model, val_loader, criterion, tgt_vocab, device, True, src_vocab, cfg.max_len, cfg.bleu_samples
        )
        print(
            f"epoch {epoch:02d} "
            f"train_ce={train_ce:.4f} train_ppl={train_ppl:.2f} "
            f"val_ce={val_ce:.4f} val_ppl={val_ppl:.2f} val_bleu={val_bleu:.2f}"
        )
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save(
                {
                    "config": asdict(cfg),
                    "model": model.state_dict(),
                    "src_itos": src_vocab.itos,
                    "tgt_itos": tgt_vocab.itos,
                },
                save_dir / "best.pt",
            )

    ckpt = torch.load(save_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_ce, test_ppl, test_bleu = evaluate(
        model, test_loader, criterion, tgt_vocab, device, True, src_vocab, cfg.max_len, cfg.bleu_samples
    )
    print(f"test_ce={test_ce:.4f} test_ppl={test_ppl:.2f} test_bleu={test_bleu:.2f}")


def build_parser(base_cfg: Config):
    p = argparse.ArgumentParser()
    for k, v in asdict(base_cfg).items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p
