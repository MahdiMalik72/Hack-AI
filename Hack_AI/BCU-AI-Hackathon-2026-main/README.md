# BCU AI Hackathon 2026 – HackAI Submission

## Team

**Team name:** HackAI

## Challenge

The goal of this project is to build a Generative AI question-answering pipeline for **100 multiple-choice questions**.

The system must:

* Load the official question file: `questions_100.csv`
* Use internet-assisted retrieval where helpful
* Use an LLM with **no more than 8B parameters**
* Produce a final CSV file with exactly these columns:

```csv
question_no,answer
```

The final answer file for our team is:

```text
HackAI_submission.csv
```

---

## Model Used

We used:

```text
Qwen2.5-7B via Ollama
```

This model is below the hackathon limit of **8B parameters**.

The model was used as part of a retrieval-augmented answering pipeline. We did not fine-tune the model.

---

## Solution Overview

Our system follows a RAG-style pipeline:

1. Load each multiple-choice question from `questions_100.csv`
2. Build multiple search queries from:

   * the question text
   * quoted phrases
   * capitalised entities
   * answer options
3. Retrieve internet evidence snippets using DuckDuckGo
4. Deduplicate repeated snippets
5. Rank evidence using keyword overlap and answer-option support
6. Build a strict prompt containing:

   * question
   * answer options
   * top-ranked evidence snippets
7. Ask Qwen2.5-7B to return only one final answer letter
8. Clean and validate the answer format
9. Export the final CSV file

The final output contains only answer letters:

```text
A, B, C, D, or E
```

Although the starter README mentioned `Unknown`, the checklist suggested final answers should be only `A–E`, so we used the safer `A–E` format for the final submission.

---

## Repository Structure

```text
BCU-AI-Hackathon-2026/
│
├── questions_100.csv
├── answer_template.csv
├── HackAI_submission.csv
├── debug_predictions.jsonl
├── search_cache.json
│
├── starter_code/
│   ├── run.py
│   ├── run_advanced.py
│   ├── run.ipynb
│   └── requirements.txt
│
├── docs/
│   └── submission_checklist.md
│
├── slides/
│   └── team_presentation_template.pptx
│
└── README.md
```

---

## Main Files

| File                            | Purpose                                                               |
| ------------------------------- | --------------------------------------------------------------------- |
| `questions_100.csv`             | Official question set                                                 |
| `HackAI_submission.csv`         | Final answer file                                                     |
| `starter_code/run.py`           | Main RAG + Ollama pipeline                                            |
| `starter_code/run_advanced.py`  | Advanced version with TF-IDF / optional embedding reranking           |
| `debug_predictions.jsonl`       | Debug log containing evidence, prompts, confidence, and model outputs |
| `search_cache.json`             | Cached DuckDuckGo search results                                      |
| `starter_code/requirements.txt` | Python dependency list                                                |

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/artidbcu/BCU-AI-Hackathon-2026.git
cd BCU-AI-Hackathon-2026
```

### 2. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install pandas ddgs openpyxl scikit-learn
```

Optional embedding support:

```powershell
python -m pip install sentence-transformers torch
```

### 3. Install Ollama

Download and install Ollama for Windows:

```text
https://ollama.com/download/windows
```

After installation, open a new PowerShell window and check:

```powershell
ollama --version
```

### 4. Pull Qwen2.5-7B

```powershell
ollama pull qwen2.5:7b
```

Test the model:

```powershell
ollama run qwen2.5:7b "Answer only A, B, C, D, or E. Question: 2+2? A. 3 B. 4 C. 5 D. 6 E. 7"
```

Expected output:

```text
B
```

---

## Running the Main Pipeline

Run a quick test on 5 questions:

```powershell
python starter_code/run.py --questions questions_100.csv --output HackAI_llm_test.csv --limit 5 --save-debug --backend ollama --model qwen2.5:7b
```

Run the full 100-question pipeline:

```powershell
python starter_code/run.py --questions questions_100.csv --output HackAI_submission.csv --save-debug --backend ollama --model qwen2.5:7b
```

This generates:

```text
HackAI_submission.csv
debug_predictions.jsonl
search_cache.json
```

---

## Running the Advanced Pipeline

The advanced pipeline includes TF-IDF reranking and optional embedding reranking.

Run with TF-IDF reranking:

```powershell
python starter_code/run_advanced.py --questions questions_100.csv --output HackAI_advanced.csv --save-debug --model qwen2.5:7b --reranker tfidf
```

Run with hybrid embedding reranking:

```powershell
python starter_code/run_advanced.py --questions questions_100.csv --output HackAI_advanced_hybrid.csv --save-debug --model qwen2.5:7b --reranker hybrid
```

The advanced version uses:

* DuckDuckGo retrieval
* TF-IDF ranking
* optional sentence-transformer embeddings
* Qwen2.5-7B answer generation
* multiple prompt styles
* majority voting
* strict answer validation

---

## Output Format

The final submission file must be named:

```text
HackAI_submission.csv
```

It must contain exactly:

```csv
question_no,answer
1,B
2,D
3,B
```

Only these answer values are used:

```text
A, B, C, D, E
```

---

## Validation

Before submission, validate the CSV:

```powershell
python -c "import pandas as pd; df=pd.read_csv('HackAI_submission.csv'); print(df.shape); print(df.head()); print(df['answer'].value_counts(dropna=False)); assert list(df.columns)==['question_no','answer']; assert len(df)==100; assert set(df['answer']).issubset(set('ABCDE')); print('HackAI_submission.csv is valid')"
```

Expected result:

```text
HackAI_submission.csv is valid
```

---

## Debugging and Error Analysis

The pipeline saves debug information to:

```text
debug_predictions.jsonl
```

This file contains:

* question number
* question text
* answer options
* search queries
* ranked evidence snippets
* heuristic answer
* LLM answer
* final answer
* confidence score

To inspect low-confidence questions:

```powershell
python -c "import json; records=[json.loads(x) for x in open('debug_predictions.jsonl',encoding='utf-8')]; weak=[r for r in records if r.get('confidence',0)<0.30 or r.get('heuristic_confidence',0)<0.30]; print(len(weak),'weak questions'); [print(r['question_no'], r.get('final_answer'), r.get('confidence', r.get('heuristic_confidence',0)), r['question'][:100]) for r in weak]"
```

---

## Techniques Used

This project uses the following techniques:

* Internet-assisted search
* Query expansion from question and answer options
* Evidence deduplication
* Evidence ranking
* TF-IDF reranking
* Optional embedding reranking
* Retrieval-Augmented Generation
* Strict answer-only prompting
* Local LLM inference with Ollama
* CSV validation
* Debug logging for answer review

---

## Reproducibility Notes

Search results can change over time because the system uses live internet retrieval through DuckDuckGo. To make reruns more reproducible, search results are cached in:

```text
search_cache.json
```

If you want to force a fresh search, delete the cache file:

```powershell
Remove-Item search_cache.json
```

Then rerun the pipeline.

---

## Final Submission Checklist

Before submitting, make sure the repository contains:

* [x] `HackAI_submission.csv`
* [x] `starter_code/run.py`
* [x] `starter_code/run_advanced.py`
* [x] `README.md`
* [x] PowerPoint or PDF presentation
* [x] Final CSV has exactly 100 rows
* [x] Final CSV columns are exactly `question_no,answer`
* [x] Final answers are only `A`, `B`, `C`, `D`, or `E`
* [x] Model name and size are documented
* [x] GitHub repository is pushed

---

## Important Note

The organiser answer key should not be submitted as part of the competition solution. It should only be used after the competition for local evaluation and error analysis.

Our submitted method uses Qwen2.5-7B through Ollama with retrieval-augmented prompting and evidence ranking.
