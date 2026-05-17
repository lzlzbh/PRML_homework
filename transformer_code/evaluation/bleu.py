from collections import Counter
from math import exp, log


def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses, references, max_n=4, weights=None):
    """
    Corpus BLEU in [0, 100].
    hypotheses / references: list of token lists (already tokenized).
    """
    weights = weights or [1.0 / max_n] * max_n
    clip_count = [0] * max_n
    total_count = [0] * max_n
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ngrams = _ngram_counts(hyp, n)
            ref_ngrams = _ngram_counts(ref, n)
            for ng, c in hyp_ngrams.items():
                total_count[n - 1] += c
                clip_count[n - 1] += min(c, ref_ngrams.get(ng, 0))

    c, r = hyp_len, ref_len
    bp = 1.0 if c > r else exp(1.0 - r / c) if c > 0 else 0.0
    terms = [
        (weights[i], clip_count[i] / total_count[i])
        for i in range(max_n)
        if total_count[i] > 0
    ]
    if not terms:
        return 0.0
    if any(p == 0 for _, p in terms):
        return 0.0
    w_sum = sum(w for w, _ in terms)
    log_bleu = sum((w / w_sum) * log(p) for w, p in terms)
    return bp * exp(log_bleu) * 100.0
