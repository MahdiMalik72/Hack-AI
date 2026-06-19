import argparse
import csv
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from urllib import request as urllib_request
from urllib.error import URLError

import pandas as pd

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_FILE = BASE_DIR.parent / "questions_100.csv"
DEFAULT_OUTPUT_FILE = "HackAI_submission.csv"
DEFAULT_DEBUG_FILE = "debug_advanced.jsonl"
DEFAULT_CACHE_FILE = "search_cache_advanced.json"

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
ALLOWED_ANSWERS = {"A", "B", "C", "D", "E"}

MAX_RESULTS_PER_QUERY = 8
MAX_QUERIES_PER_QUESTION = 5
TOP_K_EVIDENCE = 10
SEARCH_SLEEP_SECONDS = 0.5

REFERENCE_DOMAIN_HINTS = (
    "wikipedia.org",
    "britannica.com",
    ".edu",
    ".gov",
    "museum",
    "encyclopedia",
    "archive.org",
    "wikidata.org",
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "this", "to", "under",
    "was", "were", "what", "when", "where", "which", "who", "whom", "why",
    "with", "according", "mentioned", "provided", "following", "currently",
    "known", "main", "one", "original", "official", "primary", "purpose",
    "based", "used", "use", "question", "option", "options", "these", "those",
    "than", "then", "also", "about", "there", "them", "his", "her", "he",
    "she", "they", "you", "your", "i", "we", "our", "my", "me", "if", "not",
    "only", "none", "above", "all", "during", "first", "second", "third",
}


# -----------------------------
# Text utilities
# -----------------------------

def normalise_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> List[str]:
    text = normalise_text(text).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def get_option_map(row: pd.Series) -> Dict[str, str]:
    options = {}
    for letter in OPTION_LETTERS:
        value = row.get(letter, "")
        if pd.notna(value) and normalise_text(value):
            options[letter] = normalise_text(value)
    return options


def clean_answer(answer: str) -> str:
    answer = normalise_text(answer)

    match = re.search(r"\b([A-E])\b", answer.upper())
    if match:
        return match.group(1)

    first = answer.upper()[:1]
    if first in ALLOWED_ANSWERS:
        return first

    return "A"


# -----------------------------
# Data loading
# -----------------------------

def load_questions(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find questions file: {file_path}")

    df = pd.read_csv(file_path)

    required = {"question_no", "question", "A", "B", "C", "D", "E"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


# -----------------------------
# Query construction
# -----------------------------

def extract_quoted_phrases(text: str) -> List[str]:
    phrases = re.findall(r'"([^"]{2,120})"', text)
    phrases += re.findall(r"'([^']{2,120})'", text)

    cleaned = []
    for phrase in phrases:
        phrase = normalise_text(phrase)
        if phrase and phrase.lower() not in {p.lower() for p in cleaned}:
            cleaned.append(phrase)

    return cleaned[:5]


def extract_capitalised_phrases(text: str) -> List[str]:
    pattern = r"\b(?:[A-Z][\w'’.-]+(?:\s+|$)){1,7}"
    raw = re.findall(pattern, normalise_text(text))

    phrases = []
    for item in raw:
        item = normalise_text(item)
        if len(item) < 3:
            continue
        if item.lower() in STOPWORDS:
            continue
        if item.lower() not in {p.lower() for p in phrases}:
            phrases.append(item)

    phrases.sort(key=len, reverse=True)
    return phrases[:6]


def salient_terms(text: str, limit: int = 10) -> List[str]:
    counts = Counter(tokenise(text))
    ranked = sorted(counts, key=lambda x: (-counts[x], -len(x), x))
    return ranked[:limit]


def option_keywords(options: Dict[str, str], limit: int = 10) -> List[str]:
    counts = Counter()

    for value in options.values():
        counts.update(set(tokenise(value)))

    distinctive = []
    for token, count in counts.items():
        if count == 1 and len(token) >= 4:
            distinctive.append((len(token), token))

    distinctive.sort(reverse=True)
    return [token for _, token in distinctive[:limit]]


def build_search_queries(row: pd.Series, max_queries: int = MAX_QUERIES_PER_QUESTION) -> List[str]:
    question = normalise_text(row.get("question", ""))
    options = get_option_map(row)

    quoted = extract_quoted_phrases(question)
    caps = extract_capitalised_phrases(question)
    q_terms = salient_terms(question, 10)
    opt_terms = option_keywords(options, 10)

    subject = quoted[0] if quoted else caps[0] if caps else ""

    queries = []

    # 1. Best subject-focused query.
    if subject:
        queries.append(f'"{subject}" {" ".join(q_terms[:5])}')

    # 2. Wikipedia-style query often works well for factual MCQs.
    if subject:
        queries.append(f'"{subject}" wikipedia {" ".join(opt_terms[:4])}')

    # 3. Question keywords + distinctive option terms.
    queries.append(f'{" ".join(q_terms[:10])} {" ".join(opt_terms[:6])}'.strip())

    # 4. Full question with short options.
    short_options = " ".join([v for v in options.values() if len(v) <= 50])
    queries.append(f"{question} {short_options}".strip())

    # 5. Option-specific query for names/dates.
    for letter, text in options.items():
        if len(text) <= 80:
            queries.append(f'{question} "{text}"')

    deduped = []
    seen = set()

    for q in queries:
        q = normalise_text(q)[:240]
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
        if len(deduped) >= max_queries:
            break

    return deduped or [question]


# -----------------------------
# Search cache and retrieval
# -----------------------------

class SearchCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data = {}

        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, query: str):
        return self.data.get(query)

    def set(self, query: str, value):
        self.data[query] = value

    def save(self):
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def search_duckduckgo(query: str, cache: SearchCache, max_results: int = MAX_RESULTS_PER_QUERY) -> List[Dict[str, str]]:
    cached = cache.get(query)
    if cached is not None:
        return cached

    if not DDGS_AVAILABLE:
        print("[Warning] ddgs not installed.")
        return []

    evidence = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for rank, item in enumerate(results, start=1):
                evidence.append(
                    {
                        "title": normalise_text(item.get("title", "")),
                        "snippet": normalise_text(item.get("body", "")),
                        "url": normalise_text(item.get("href", "")),
                        "query": query,
                        "rank": rank,
                    }
                )
    except Exception as error:
        print(f"[Warning] search failed for query={query!r}: {error}")

    cache.set(query, evidence)
    time.sleep(SEARCH_SLEEP_SECONDS)
    return evidence


def deduplicate_evidence(evidence: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    output = []

    for item in evidence:
        text = normalise_text(item.get("title", "") + " " + item.get("snippet", ""))
        key = re.sub(r"\W+", " ", text.lower()).strip()[:300]

        if not key:
            key = normalise_text(item.get("url", "")).lower()

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


# -----------------------------
# Ranking: TF-IDF + optional embeddings
# -----------------------------

def domain_bonus(url: str) -> float:
    url = normalise_text(url).lower()
    bonus = 0.0

    if any(h in url for h in REFERENCE_DOMAIN_HINTS):
        bonus += 0.15

    if "wikipedia.org" in url:
        bonus += 0.20

    if url.endswith(".pdf"):
        bonus += 0.05

    return bonus


def option_support_scores(options: Dict[str, str], text: str) -> Dict[str, float]:
    text_norm = normalise_text(text)
    text_lower = text_norm.lower()
    text_tokens = set(tokenise(text_norm))

    scores = {}

    for letter, option_text in options.items():
        option_tokens = set(tokenise(option_text))
        overlap = len(option_tokens & text_tokens)
        coverage = overlap / max(1.0, len(option_tokens))

        exact_bonus = 0.0
        if option_text.lower() in text_lower:
            exact_bonus += 1.5

        long_token_bonus = 0.0
        for tok in option_tokens:
            if len(tok) >= 5 and tok in text_tokens:
                long_token_bonus += 0.12

        scores[letter] = 2.5 * coverage + exact_bonus + long_token_bonus

    return scores


def tfidf_scores(query_text: str, docs: List[str]) -> List[float]:
    if not docs:
        return []

    if not SKLEARN_AVAILABLE:
        q_tokens = set(tokenise(query_text))
        values = []
        for doc in docs:
            d_tokens = set(tokenise(doc))
            values.append(len(q_tokens & d_tokens) / max(1.0, len(q_tokens | d_tokens)))
        return values

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )

    matrix = vectorizer.fit_transform([query_text] + docs)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [float(x) for x in sims]


def embedding_scores(query_text: str, docs: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    if not EMBEDDINGS_AVAILABLE or not docs:
        return [0.0 for _ in docs]

    model = SentenceTransformer(model_name)
    vectors = model.encode([query_text] + docs, normalize_embeddings=True)
    query_vec = vectors[0]
    doc_vecs = vectors[1:]
    return [float(query_vec @ v) for v in doc_vecs]


def rank_evidence(
    row: pd.Series,
    evidence: List[Dict[str, str]],
    top_k: int = TOP_K_EVIDENCE,
    reranker: str = "tfidf",
) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    question = normalise_text(row.get("question", ""))
    options = get_option_map(row)

    evidence = deduplicate_evidence(evidence)
    docs = [
        normalise_text(item.get("title", "") + " " + item.get("snippet", ""))
        for item in evidence
    ]

    option_text = " ".join(options.values())
    query_text = f"{question} {option_text}"

    tfidf = tfidf_scores(query_text, docs)

    if reranker in {"embedding", "hybrid"}:
        emb = embedding_scores(query_text, docs)
    else:
        emb = [0.0 for _ in docs]

    aggregate_option_scores = {letter: 0.0 for letter in options}
    ranked = []

    for i, item in enumerate(evidence):
        doc = docs[i]
        opt_scores = option_support_scores(options, doc)

        for letter, score in opt_scores.items():
            aggregate_option_scores[letter] += score

        best_option_support = max(opt_scores.values()) if opt_scores else 0.0

        if reranker == "embedding":
            retrieval_score = emb[i]
        elif reranker == "hybrid":
            retrieval_score = 0.65 * tfidf[i] + 0.35 * emb[i]
        else:
            retrieval_score = tfidf[i]

        score = (
            3.0 * retrieval_score
            + 0.6 * best_option_support
            + domain_bonus(item.get("url", ""))
            + (0.05 / max(1, int(item.get("rank", 1))))
        )

        ranked.append(
            {
                **item,
                "score": round(score, 4),
                "tfidf": round(tfidf[i], 4),
                "embedding": round(emb[i], 4),
                "best_option_support": round(best_option_support, 4),
                "option_support": {k: round(v, 4) for k, v in opt_scores.items()},
            }
        )

    ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    return ranked[:top_k], aggregate_option_scores


# -----------------------------
# Prompting and Ollama
# -----------------------------

def build_prompt(row: pd.Series, ranked_evidence: List[Dict[str, str]], style: str = "normal") -> str:
    question = normalise_text(row.get("question", ""))
    options = get_option_map(row)

    option_lines = "\n".join([f"{k}. {v}" for k, v in options.items()])

    evidence_lines = []
    for i, item in enumerate(ranked_evidence, start=1):
        evidence_lines.append(
            f"[{i}] {item.get('title', '')}\n"
            f"Snippet: {item.get('snippet', '')}\n"
            f"URL: {item.get('url', '')}"
        )

    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "No useful evidence retrieved."

    if style == "strict":
        instruction = """
You are a careful question-answering system.
Use ONLY the evidence snippets.
Select the answer option best supported by the evidence.
Return exactly one character: A, B, C, D, or E.
Do not explain.
""".strip()
    elif style == "option_elimination":
        instruction = """
You are solving a multiple-choice factual question.
Compare each option against the evidence.
Eliminate unsupported options.
Return exactly one character only: A, B, C, D, or E.
Do not explain.
""".strip()
    else:
        instruction = """
Answer the multiple-choice question using the evidence.
Return only one final letter: A, B, C, D, or E.
Do not include words or explanation.
""".strip()

    return f"""
{instruction}

Question:
{question}

Options:
{option_lines}

Evidence:
{evidence_text}

Final answer:
""".strip()


def call_ollama(prompt: str, model: str, host: str = "http://localhost:11434") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 6,
            "top_p": 0.9,
        },
    }

    req = urllib_request.Request(
        url=f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=240) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "")
    except URLError as error:
        raise RuntimeError(f"Could not reach Ollama: {error}") from error


def heuristic_answer(
    row: pd.Series,
    ranked_evidence: List[Dict[str, str]],
    aggregate_option_scores: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    options = get_option_map(row)
    scores = {letter: 0.0 for letter in options}

    for rank, item in enumerate(ranked_evidence, start=1):
        weight = 1.0 / math.sqrt(rank)
        support = item.get("option_support", {})

        for letter in options:
            scores[letter] += float(support.get(letter, 0.0)) * weight

    for letter, score in aggregate_option_scores.items():
        scores[letter] += 0.15 * float(score)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    total = sum(max(v, 0.0) for v in scores.values()) + 1e-9

    meta = {
        "heuristic_best_score": round(best_score, 4),
        "heuristic_second_score": round(second_score, 4),
        "heuristic_margin": round(best_score - second_score, 4),
        "heuristic_confidence": round(best_score / total, 4),
    }

    for letter, score in scores.items():
        meta[f"heuristic_score_{letter}"] = round(score, 4)

    return best, meta


def llm_vote(row: pd.Series, ranked_evidence: List[Dict[str, str]], model: str) -> Tuple[str, Dict]:
    styles = ["normal", "strict", "option_elimination"]

    votes = []
    raw_outputs = []

    for style in styles:
        prompt = build_prompt(row, ranked_evidence, style=style)
        raw = call_ollama(prompt, model=model)
        ans = clean_answer(raw)
        raw_outputs.append({"style": style, "raw": raw, "answer": ans})
        votes.append(ans)

    counts = Counter(votes)
    answer, count = counts.most_common(1)[0]

    return answer, {
        "llm_votes": dict(counts),
        "llm_vote_count": count,
        "llm_raw_outputs": raw_outputs,
    }


def combined_answer(
    row: pd.Series,
    ranked_evidence: List[Dict[str, str]],
    aggregate_option_scores: Dict[str, float],
    model: str,
) -> Tuple[str, Dict]:
    heuristic, hmeta = heuristic_answer(row, ranked_evidence, aggregate_option_scores)
    llm, lmeta = llm_vote(row, ranked_evidence, model=model)

    # Practical ensemble:
    # - If LLM has majority agreement, trust LLM.
    # - If LLM split badly, fall back to heuristic.
    # - If heuristic confidence is very strong, trust heuristic.
    vote_count = int(lmeta.get("llm_vote_count", 0))
    hconf = float(hmeta.get("heuristic_confidence", 0.0))

    if hconf >= 0.75:
        final = heuristic
        source = "heuristic_high_confidence"
    elif vote_count >= 2:
        final = llm
        source = "llm_majority"
    else:
        final = heuristic
        source = "heuristic_fallback"

    meta = {
        **hmeta,
        **lmeta,
        "heuristic_answer": heuristic,
        "llm_answer": llm,
        "decision_source": source,
    }

    return clean_answer(final), meta


# -----------------------------
# Pipeline
# -----------------------------

def answer_question(
    row: pd.Series,
    cache: SearchCache,
    model: str,
    reranker: str,
) -> Tuple[Dict[str, str], Dict]:
    queries = build_search_queries(row)

    evidence = []
    for query in queries:
        evidence.extend(search_duckduckgo(query, cache=cache))

    ranked_evidence, aggregate_option_scores = rank_evidence(
        row=row,
        evidence=evidence,
        top_k=TOP_K_EVIDENCE,
        reranker=reranker,
    )

    answer, meta = combined_answer(
        row=row,
        ranked_evidence=ranked_evidence,
        aggregate_option_scores=aggregate_option_scores,
        model=model,
    )

    prediction = {
        "question_no": row.get("question_no"),
        "answer": answer,
    }

    debug = {
        "question_no": row.get("question_no"),
        "question": normalise_text(row.get("question", "")),
        "options": get_option_map(row),
        "queries": queries,
        "ranked_evidence": ranked_evidence,
        "aggregate_option_scores": {
            k: round(v, 4) for k, v in aggregate_option_scores.items()
        },
        "final_answer": answer,
        **meta,
    }

    return prediction, debug


def validate_submission(df: pd.DataFrame) -> None:
    if list(df.columns) != ["question_no", "answer"]:
        raise ValueError("CSV must have columns: question_no,answer")

    invalid = sorted(set(df["answer"]) - ALLOWED_ANSWERS)
    if invalid:
        raise ValueError(f"Invalid answer values: {invalid}")


def write_debug(records: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    questions = load_questions(args.questions)

    if args.limit:
        questions = questions.head(args.limit)

    cache = SearchCache(args.cache_file)

    predictions = []
    debug_records = []

    for idx, row in questions.iterrows():
        qno = row.get("question_no", idx + 1)
        print(f"[{idx + 1}/{len(questions)}] Question {qno}")

        prediction, debug = answer_question(
            row=row,
            cache=cache,
            model=args.model,
            reranker=args.reranker,
        )

        predictions.append(prediction)
        debug_records.append(debug)

        print(
            f"    Final={debug['final_answer']} | "
            f"LLM={debug.get('llm_answer')} | "
            f"Heuristic={debug.get('heuristic_answer')} | "
            f"Source={debug.get('decision_source')}"
        )

    df = pd.DataFrame(predictions, columns=["question_no", "answer"])
    validate_submission(df)

    df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    cache.save()

    if args.save_debug:
        write_debug(debug_records, args.debug_file)
        print(f"Saved debug to {args.debug_file}")

    print(f"Saved output to {args.output}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced HackAI QA pipeline")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_FILE))
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--reranker", choices=["tfidf", "embedding", "hybrid"], default="tfidf")
    parser.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--debug-file", default=DEFAULT_DEBUG_FILE)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())