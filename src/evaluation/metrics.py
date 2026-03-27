import math
import re
import string
from collections import Counter, defaultdict
from typing import Iterable

ARTICLES = {"a", "an", "the"}
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(PUNCT_TRANSLATION)
    text = re.sub(r"\s+", " ", text)
    return text


def normalized_tokens(text: str, *, drop_articles: bool = False) -> list[str]:
    tokens = [token for token in normalize_text(text).split() if token]
    if drop_articles:
        tokens = [token for token in tokens if token not in ARTICLES]
    return tokens


def strict_exact_match(prediction: str, references: list[str]) -> float:
    return float(any(prediction.strip() == reference.strip() for reference in references))


def top1_accuracy(prediction: str, references: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    return float(any(normalized_prediction == normalize_text(reference) for reference in references))


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_4(prediction: str, references: list[str]) -> float:
    pred_tokens = normalized_tokens(prediction)
    ref_tokens_list = [normalized_tokens(reference) for reference in references if reference.strip()]

    if not pred_tokens or not ref_tokens_list:
        return 0.0

    precisions: list[float] = []
    for n in range(1, 5):
        pred_ngrams = Counter(_ngrams(pred_tokens, n))
        if not pred_ngrams:
            precisions.append(0.0)
            continue

        max_ref_counts: Counter[tuple[str, ...]] = Counter()
        for ref_tokens in ref_tokens_list:
            ref_counts = Counter(_ngrams(ref_tokens, n))
            for gram, count in ref_counts.items():
                if count > max_ref_counts[gram]:
                    max_ref_counts[gram] = count

        clipped = sum(min(count, max_ref_counts[gram]) for gram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append((clipped + 1.0) / (total + 1.0))

    pred_len = len(pred_tokens)
    ref_lens = [len(tokens) for tokens in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - pred_len), ref_len))

    if pred_len == 0:
        brevity_penalty = 0.0
    elif pred_len > closest_ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - (closest_ref_len / pred_len))

    score = brevity_penalty * math.exp(sum(0.25 * math.log(max(p, 1e-12)) for p in precisions))
    return max(0.0, min(1.0, score))


def meteor(prediction: str, references: list[str], alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    """
    Lightweight METEOR implementation based on exact unigram matches and the
    standard fragmentation penalty formula.
    """
    pred_tokens = normalized_tokens(prediction)
    if not pred_tokens or not references:
        return 0.0

    best_score = 0.0
    for reference in references:
        ref_tokens = normalized_tokens(reference)
        if not ref_tokens:
            continue

        ref_positions: defaultdict[str, list[int]] = defaultdict(list)
        for idx, token in enumerate(ref_tokens):
            ref_positions[token].append(idx)

        matches: list[tuple[int, int]] = []
        used_ref_indices: set[int] = set()
        last_ref_idx = -1

        for pred_idx, token in enumerate(pred_tokens):
            candidates = [i for i in ref_positions.get(token, []) if i not in used_ref_indices and i > last_ref_idx]
            if not candidates:
                candidates = [i for i in ref_positions.get(token, []) if i not in used_ref_indices]
            if not candidates:
                continue
            ref_idx = candidates[0]
            used_ref_indices.add(ref_idx)
            last_ref_idx = ref_idx
            matches.append((pred_idx, ref_idx))

        m = len(matches)
        if m == 0:
            continue

        precision = m / len(pred_tokens)
        recall = m / len(ref_tokens)
        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

        chunks = 1
        for i in range(1, len(matches)):
            prev_pred, prev_ref = matches[i - 1]
            curr_pred, curr_ref = matches[i]
            if curr_pred != prev_pred + 1 or curr_ref != prev_ref + 1:
                chunks += 1

        penalty = gamma * ((chunks / m) ** beta)
        score = f_mean * (1 - penalty)
        best_score = max(best_score, score)

    return max(0.0, min(1.0, best_score))


def cider(prediction: str, references: list[str], max_n: int = 4, sigma: float = 6.0) -> float:
    """
    CIDEr-style score using TF-IDF weighted n-gram cosine similarity.
    Returned on the common 0-10 CIDEr scale.
    """
    pred_tokens = normalized_tokens(prediction)
    ref_tokens_list = [normalized_tokens(reference) for reference in references if reference.strip()]

    if not pred_tokens or not ref_tokens_list:
        return 0.0

    num_refs = len(ref_tokens_list)
    document_frequency: dict[tuple[int, tuple[str, ...]], int] = defaultdict(int)
    for ref_tokens in ref_tokens_list:
        seen: set[tuple[int, tuple[str, ...]]] = set()
        for n in range(1, max_n + 1):
            for gram in set(_ngrams(ref_tokens, n)):
                key = (n, gram)
                if key not in seen:
                    document_frequency[key] += 1
                    seen.add(key)

    def tfidf_vector(tokens: list[str], n: int) -> dict[tuple[str, ...], float]:
        grams = _ngrams(tokens, n)
        counts = Counter(grams)
        total = sum(counts.values())
        if total == 0:
            return {}
        vector: dict[tuple[str, ...], float] = {}
        for gram, count in counts.items():
            df = document_frequency.get((n, gram), 0)
            idf = math.log((num_refs + 1.0) / (df + 1.0))
            vector[gram] = (count / total) * idf
        return vector

    def cosine_similarity(vec_a: dict[tuple[str, ...], float], vec_b: dict[tuple[str, ...], float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    pred_len = len(pred_tokens)
    scores: list[float] = []
    for ref_tokens in ref_tokens_list:
        ref_len = len(ref_tokens)
        length_penalty = math.exp(-((pred_len - ref_len) ** 2) / (2 * sigma * sigma))
        n_scores = []
        for n in range(1, max_n + 1):
            pred_vec = tfidf_vector(pred_tokens, n)
            ref_vec = tfidf_vector(ref_tokens, n)
            n_scores.append(cosine_similarity(pred_vec, ref_vec) * length_penalty)
        scores.append(sum(n_scores) / max_n)

    return max(0.0, 10.0 * sum(scores) / len(scores))


def summarize_metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    values = [float(value) for value in values]
    values.sort()
    midpoint = len(values) // 2
    median = values[midpoint] if len(values) % 2 == 1 else (values[midpoint - 1] + values[midpoint]) / 2

    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(median, 4),
        "min": round(values[0], 4),
        "max": round(values[-1], 4),
    }


def ensure_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if isinstance(item, str) and item.strip()]