# 📊 RAG Evaluation Metrics — Measuring Retrieval Quality with DeepEval

Building a RAG system is one thing. **Proving it actually works is another.** This project builds a simple, real RAG pipeline (Chroma + OpenAI), then rigorously evaluates it using **DeepEval's** LLM-judged metrics — Contextual Precision, Contextual Recall, and Contextual Relevancy — deliberately corrupting the retrieved context in each test to prove the metrics actually detect bad retrieval rather than just rubber-stamping every result.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-1C3C3C?logo=langchain&logoColor=white)
![DeepEval](https://img.shields.io/badge/DeepEval-LLM--as--Judge-8A2BE2)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00)

---

## 📌 Overview

Most RAG tutorials stop at "it retrieved something and the LLM answered." This notebook asks the harder question: **how do you know the retrieval was actually good?** It builds a working RAG chain, then wraps it in a formal evaluation framework using **LLM-as-judge metrics** — each one tested against both clean and deliberately noisy context, so the metric's behaviour under failure is verified, not just assumed.

---

## 🏗️ The Complete Pipeline

```
CSV Knowledge Base (rag_eval_docs.csv)
        │
        ▼
LangChain Document objects ──► OpenAI Embeddings ──► Chroma (persisted, cosine)
        │
        ▼
Similarity-Score-Threshold Retriever (k=3, threshold=0.3)
        │
        ▼
RAG Chain with Sources (retrieval + generation + returned context)
        │
        ▼
DeepEval Test Cases ──► Contextual Precision / Recall / Relevancy
        │
        ▼
   Score + Success/Fail + LLM-generated Reason
```

---

## 🔬 Part 1 — Building a Real (If Simple) RAG System

### Retrieval with a Score Threshold
```python
similarity_retriever = chroma_db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={"k": 3, "score_threshold": 0.3}
)
```
Unlike a plain top-k retriever (which always returns *something*, as demonstrated in earlier retrieval work), `search_type='similarity_score_threshold'` **rejects weak matches outright** — only chunks scoring above 0.3 similarity are returned at all. This is the fix for the "biryani problem": a retriever that can genuinely return nothing when nothing relevant exists.

### RAG Chain That Returns Its Own Sources
```python
rag_chain_w_sources = (
    {"context": similarity_retriever, "question": RunnablePassthrough()}
    | RunnablePassthrough.assign(response=src_rag_response_chain)
)
```
The chain returns **both** the generated answer *and* the raw retrieved documents in one object — essential for evaluation, since you can't grade retrieval quality without seeing exactly what was retrieved.

---

## 🔬 Part 2 — DeepEval: LLM-as-Judge Evaluation

**DeepEval** wraps a test case — question, actual answer, expected (ground-truth) answer, and the retrieved context — and scores it using **another LLM as the judge** (`gpt-4o` here), returning a numeric score, a pass/fail verdict against a threshold, and a written reason.

```python
test_case = LLMTestCase(
    input=response['question'],
    actual_output=response['response'],
    expected_output=human_answer,        # the ground-truth benchmark
    retrieval_context=new_context
)
```

### Metric 1 — Contextual Precision
> *Of the context that was retrieved, how much of it was actually relevant?*

```python
metrics = ContextualPrecisionMetric(threshold=0.50, model="gpt-4o", include_reason=True)
```
Tested by **deliberately injecting irrelevant chunks** ("Machine Learning is the study of algorithms...") alongside the genuinely retrieved context — precision should drop when noise is added, proving the metric actually penalizes irrelevant retrieval rather than just checking the answer sounds right.

### Metric 2 — Contextual Recall
> *Of everything needed to answer correctly, how much did retrieval actually find?*

```python
metrics = ContextualRecallMetric(threshold=0.60, model="gpt-4o", include_reason=True)
```
Tested with **two parallel test cases** — one using the real retrieved context, one using context that's *completely unrelated* to the ground-truth answer ("NVIDIA makes chips for AI"). The two results are compared directly to confirm recall correctly collapses when the right information genuinely isn't present.

### Metric 3 — Contextual Relevancy
> *Overall, how relevant is the retrieved context to the question, independent of the final answer?*

```python
metrics = ContextualRelevancyMetric(threshold=0.80, model="gpt-4o", include_reason=True)
```
Tested against a context list padded with plausible-sounding but off-topic sentences ("Google and Microsoft are battling out the market share for AI Chatbots") mixed in with genuine content — relevancy scoring should reflect that dilution.

---

## 💡 Why Corrupt the Context Deliberately?

This is the notebook's most important instinct, worth calling out explicitly: **every metric is tested against a scenario designed to make it fail.** A metric that always scores high regardless of input is worthless — proving each metric responds correctly to *bad* context is what makes trusting a *good* score meaningful. This is the same evaluation discipline that separates a benchmarked RAG system from one that merely "seems to work."

---

## 🗂️ Repository Structure

```
rag-evaluation-metrics-deepeval/
├── GenAI_24_RAGEvalMatrix_01.ipynb   # Main notebook
├── requirements.txt                   # Dependencies
├── .gitignore                         # Keeps secrets, data & DB out of git
├── .env.example                       # Template for required environment variables
└── README.md                          # This documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) — required for both the RAG pipeline **and** DeepEval's LLM-as-judge scoring (which itself calls `gpt-4o`)
- `rag_eval_docs.csv` — the source knowledge base (the notebook auto-downloads it via `gdown`, with a manual Google Drive fallback link)

### Installation

```bash
git clone https://github.com/Kailaswadje/rag-evaluation-metrics-deepeval.git
cd rag-evaluation-metrics-deepeval

pip install -r requirements.txt

cp .env.example .env
# add your OPENAI_API_KEY to .env

jupyter notebook GenAI_24_RAGEvalMatrix_01.ipynb
```

> ⚠️ DeepEval's metrics make **additional LLM calls per test case** (the judge model) on top of your RAG pipeline's own calls — expect higher token usage than a typical RAG demo. Clear notebook outputs before pushing.

---

## 🧠 Key Takeaways

- **A RAG system without evaluation is a guess wearing a demo** — `similarity_score_threshold` retrieval and DeepEval scoring together turn "it seems to work" into a measurable, reproducible claim
- **LLM-as-judge is now a standard evaluation pattern** — using `gpt-4o` to score another model's retrieval and generation quality, with a written justification for every score
- **Testing a metric against corrupted input is how you trust it on clean input** — if Contextual Precision can't detect injected noise, its high score on real retrieval means nothing
- **Precision, Recall, and Relevancy measure different failure modes** — precision catches irrelevant retrieval, recall catches *missing* retrieval, relevancy measures overall context-question alignment independent of the final answer
- This evaluation methodology is the direct blueprint for benchmarking retrieval quality (MRR and beyond) in my dissertation's hybrid GraphRAG + HippoRAG platform

---

## 🔮 Possible Extensions

- [ ] Add **Faithfulness** and **Answer Relevancy** metrics to evaluate the generation side, not just retrieval
- [ ] Run the full metric suite across a batch of test questions and aggregate scores into a benchmark table
- [ ] Compare metric scores across different `score_threshold` and `k` retriever settings
- [ ] Log results to DeepEval's dashboard for tracking evaluation runs over time

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If this clarified how to actually evaluate a RAG system, consider giving it a star!
