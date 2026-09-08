# 📊 RAG Evaluation Suite — Retrieval & Generation Metrics with DeepEval

Building a RAG system is one thing. **Proving it actually works — both what it retrieves and what it generates — is another.** This project builds a simple, real RAG pipeline (Chroma + OpenAI), then evaluates it end to end using **DeepEval's** LLM-as-judge metrics: three **retrieval-side** metrics (Contextual Precision, Recall, Relevancy) and four **generation-side** metrics (Answer Relevancy, Faithfulness, Hallucination, and a fully custom GEval "Fact Checker"). Every metric is deliberately tested against corrupted or wrong input to prove it actually detects failure, not just rubber-stamps good-looking answers.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-1C3C3C?logo=langchain&logoColor=white)
![DeepEval](https://img.shields.io/badge/DeepEval-LLM--as--Judge-8A2BE2)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00)

---

## 📌 Overview

Most RAG tutorials stop at "it retrieved something and the LLM answered." This notebook asks the harder question, twice over: **was the retrieval good, and was the generated answer actually trustworthy?** It builds a working RAG chain, then wraps it in a **full seven-metric evaluation suite** spanning both halves of the pipeline — retrieval quality and generation quality are different failure modes, and this project measures each one deliberately.

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
        ├──► RETRIEVAL METRICS: Contextual Precision / Recall / Relevancy
        │
        └──► GENERATOR METRICS: Answer Relevancy / Faithfulness / Hallucination / Custom GEval
                        │
                        ▼
            Score + Success/Fail + LLM-generated Reason (per metric)
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
Unlike a plain top-k retriever, `search_type='similarity_score_threshold'` **rejects weak matches outright** — only chunks scoring above 0.3 similarity are returned at all, a genuine fix for retrievers that always return *something* regardless of relevance.

### RAG Chain That Returns Its Own Sources
```python
rag_chain_w_sources = (
    {"context": similarity_retriever, "question": RunnablePassthrough()}
    | RunnablePassthrough.assign(response=src_rag_response_chain)
)
```
The chain returns **both** the generated answer *and* the raw retrieved documents in one object — essential, since neither retrieval nor generation metrics can be scored without seeing exactly what was retrieved and what was said.

---

## 🔬 Part 2 — Retrieval-Side Metrics (Is the Context Good?)

| Metric | Question It Answers | Corrupted-Input Test |
|---|---|---|
| **Contextual Precision** | Of what was retrieved, how much was relevant? | Irrelevant chunks injected alongside real context |
| **Contextual Recall** | Of what's needed, how much was retrieved? | Compared against context entirely unrelated to the ground truth |
| **Contextual Relevancy** | How relevant is the retrieved context overall? | Real context diluted with plausible-sounding off-topic sentences |

Each metric is instantiated with `LLMTestCase(input, actual_output, expected_output, retrieval_context)` and scored by an LLM judge (`gpt-4o`) against a threshold, returning a numeric score, pass/fail, and a written reason.

---

## 🔬 Part 3 — Generator Metrics (Is the Answer Trustworthy?) — *New in this version*

This is the notebook's newest and most important addition: evaluating the **generated answer itself**, not just what fed into it.

### Answer Relevancy
```python
test_case = LLMTestCase(input=response['question'], actual_output=response['response'])
metrics = AnswerRelevancyMetric(threshold=0.50, model="gpt-4o-mini", include_reason=True)
```
> *Does the answer actually address the question asked?* No context involved here — purely question-vs-answer alignment.

### Faithfulness
```python
metrics = FaithfulnessMetric(threshold=0.50, model="gpt-4o-mini", include_reason=True)
```
> *Does the answer stay true to the retrieved context, without adding unsupported claims?* This is scored against `retrieval_context`, catching answers that drift beyond what was actually retrieved.

### Hallucination Check — Tested Both Ways
```python
human_ground_truth_context = ["Artificial intelligence refers to machines mimicking human intelligence..."]

# Test 1: the model's real (reasonable) response
metrics = HallucinationMetric(threshold=0.10, model="gpt-4o-mini", include_reason=True)

# Test 2: a deliberately fabricated response
ai_response = "AI refers to machines mimicking human intelligence to produces neurons, atoms, electrodes, bio chemicals"
```
The second test case swaps in a **deliberately nonsensical, technically-fake-sounding answer** — the kind of confident-but-wrong output a hallucinating model might produce — and re-scores it against the *same* ground-truth context. This is the same "test the metric against a failure case" discipline used for the retrieval metrics, now applied to hallucination detection specifically: if the metric can't catch *this* fabrication, it can't be trusted on subtler ones either.

### Custom LLM-as-Judge — `GEval` "RAG Fact Checker"
```python
metric = GEval(
    threshold=0.5, model="gpt-4o-mini", name="RAG Fact Checker",
    evaluation_steps=[
        "Create a list of statements from 'actual output'",
        "Validate if they are relevant and answer the given question in 'input', penalize if irrelevant",
        "Validate if they exist in 'expected output', penalize if missing or factually wrong",
        "Validate if these statements are grounded in the 'retrieval context', penalize if unsupported",
        "Penalize if any statements seem invented or made up given the input and retrieval context"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                       LLMTestCaseParams.EXPECTED_OUTPUT]
)
```
Beyond DeepEval's built-in metrics, `GEval` allows **defining a fully custom evaluation rubric in plain English** — here, a five-step fact-checking procedure that decomposes the answer into statements and checks each one for relevance, correctness, grounding, and fabrication. This is the most flexible metric in the suite: any evaluation criterion expressible as a checklist can become a scored, LLM-judged metric.

---

## 💡 Why Corrupt the Input Deliberately, Every Single Time?

This is the notebook's defining discipline, and it now runs through **all seven metrics**, not just the retrieval three: **every metric is tested against a scenario designed to make it fail.** A metric that always scores high regardless of input is worthless. Proving each metric responds correctly to *bad* context, *bad* answers, and outright *fabrication* is what makes trusting a *good* score meaningful — this is the difference between a benchmarked RAG system and one that merely "seems to work."

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
- An [OpenAI API key](https://platform.openai.com/api-keys) — required for the RAG pipeline **and** every DeepEval metric's LLM-as-judge scoring (`gpt-4o` / `gpt-4o-mini`)
- `rag_eval_docs.csv` — the source knowledge base (auto-downloaded via `gdown`, with a manual Google Drive fallback link in the notebook)

### Installation

```bash
git clone https://github.com/Kailaswadje/rag-evaluation-metrics-deepeval.git
cd rag-evaluation-metrics-deepeval

pip install -r requirements.txt

cp .env.example .env
# add your OPENAI_API_KEY to .env

jupyter notebook GenAI_24_RAGEvalMatrix_01.ipynb
```

> ⚠️ With **seven metrics**, each making its own judge-model LLM call — some run twice for the corrupted-input comparisons — this notebook makes noticeably more API calls than a typical RAG demo. Expect proportionally higher token usage. Clear notebook outputs before pushing.

---

## 🧠 Key Takeaways

- **Retrieval quality and generation quality are separate failure modes** — a RAG system can retrieve perfectly and still hallucinate, or retrieve poorly and still produce a plausible-sounding wrong answer; this suite catches both independently
- **Faithfulness and Hallucination are related but distinct** — faithfulness asks whether the answer stays within the retrieved context; hallucination checks the answer against a broader ground truth, catching fabrication even when context handling looks fine
- **Testing a metric against a deliberately fabricated answer** ("neurons, atoms, electrodes, bio chemicals") is the sharpest version of this notebook's core discipline — if a metric can't flag *obvious* nonsense, it can't be trusted on subtle drift
- **`GEval` turns any written evaluation checklist into a scored metric** — the "RAG Fact Checker" shows that custom, domain-specific evaluation criteria don't require building a metric from scratch
- This seven-metric retrieval-and-generation evaluation suite is the direct blueprint for benchmarking both retrieval quality (MRR and beyond) and answer trustworthiness in my dissertation's hybrid GraphRAG + HippoRAG platform

---

## 🔮 Possible Extensions

- [ ] Run the full seven-metric suite across a batch of test questions and aggregate results into a benchmark table
- [ ] Compare Faithfulness and Hallucination scores across different retriever `k` and `score_threshold` settings
- [ ] Add a second `GEval` custom metric for a different evaluation angle (e.g., conciseness, tone)
- [ ] Log results to DeepEval's dashboard for tracking evaluation runs over time

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If this full evaluation suite helped you trust your RAG system's output, consider giving it a star!
