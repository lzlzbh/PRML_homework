import argparse
from pathlib import Path

import torch

from train import ROOT, Config, Transformer, greedy_decode, tokenize


class LoadedVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad_idx = self.stoi["<pad>"]
        self.bos_idx = self.stoi["<bos>"]
        self.eos_idx = self.stoi["<eos>"]
        self.unk_idx = self.stoi["<unk>"]

    def encode(self, text):
        return [self.bos_idx] + [
            self.stoi.get(tok, self.unk_idx) for tok in tokenize(text)
        ] + [self.eos_idx]

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.itos[int(i)]
            if w == "<eos>":
                break
            if w not in ("<pad>", "<bos>"):
                words.append(w)
        return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "scratch_transformer" / "best.pt"),
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--max_len", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    cfg = Config(**ckpt["config"])
    src_vocab = LoadedVocab(ckpt["src_itos"])
    tgt_vocab = LoadedVocab(ckpt["tgt_itos"])
    model = Transformer(cfg, len(src_vocab.itos), len(tgt_vocab.itos), src_vocab.pad_idx)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    src = torch.tensor(src_vocab.encode(args.text), dtype=torch.long)
    print(" ".join(greedy_decode(model, src, src_vocab, tgt_vocab, args.max_len, device)))


if __name__ == "__main__":
    main()
