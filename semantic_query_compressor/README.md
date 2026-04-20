# Semantic Query Compressor

This project compresses a long text query into a shorter, generalized query and checks whether a Large Language Model responds consistently to both versions.

It produces the four core outputs requested:

- `compressed_query`: the shorter query created from the original text.
- `original_output`: the Groq LLM response to the original long query.
- `compressed_output`: the Groq LLM response to the compressed query.
- `similarity_score`: a score from `0` to `1` comparing the two LLM responses.

## Approach

This version uses a lightweight, transparent graph algorithm before adding heavier semantic models.

1. **Compression**
   The compressor uses AMR-inspired graph compression with intent preservation. It splits the query into sentences, extracts content words as lightweight AMR-like concepts, builds a concept graph from adjacency and co-occurrence relations, and scores sentences by central concept coverage. It always preserves detected task and constraint sentences, then uses Maximal Marginal Relevance to add high-value, non-redundant context sentences.

   You can also choose LLM-based compression. In that mode, the configured Groq model compresses the original query before the system evaluates whether the original and compressed prompts produce similar LLM responses.

2. **LLM Evaluation**
   The pipeline sends both the original query and compressed query to Groq using the same model, temperature, system prompt, and generation settings. Keeping those settings identical makes the comparison more meaningful.

3. **Similarity Scoring**
   The current scorer uses TF-IDF cosine similarity between `original_output` and `compressed_output`. This is a standard lexical semantic proxy and returns a bounded score between `0` and `1`.

Later, you can replace the lightweight AMR-inspired compressor with a full AMR parser, sentence embeddings, abstractive summarization, or an LLM-based summarizer.

## File Structure

```text
semantic_query_compressor/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- requirements-dev.txt
|-- .env.example
|-- example.txt
|-- main.py
|-- src/
|   `-- semantic_query_compressor/
|       |-- __init__.py
|       |-- cli.py
|       |-- compressor.py
|       |-- llm.py
|       |-- pipeline.py
|       |-- similarity.py
|       `-- text_utils.py
`-- tests/
    `-- test_pipeline.py
```

## Requirements

- Python 3.10+
- A Groq API key

## Setup

From the project directory:

```bash
cd semantic_query_compressor
python -m venv .venv
```

Activate the virtual environment.

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Create a `.env` file:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

You can change `GROQ_MODEL` to another Groq chat model if needed.

## Python Usage

```python
from semantic_query_compressor import compress_and_evaluate

query = """
Paste a long text query here. It should contain at least 100 words because the
compressor is designed for long prompts where sentence-level compression is useful.
"""

result = compress_and_evaluate(query, compression_method="amr")

print(result.compressed_query)
print(result.original_output)
print(result.compressed_output)
print(result.similarity_score)
print(result.compression_method)
```

The function returns a `CompressionResult` dataclass:

```python
CompressionResult(
    compressed_query="...",
    original_output="...",
    compressed_output="...",
    similarity_score=0.82,
    compression_method="amr",
)
```

To get a dictionary:

```python
result.as_dict()
```

## CLI Usage

Pass the query directly:

```bash
semantic-query-compressor --query "Paste a long query with more than 100 words here..." --pretty
```

Or read from a file:

```bash
semantic-query-compressor --file example.txt --pretty
```

You can tune how aggressive the compression is:

```bash
semantic-query-compressor --file example.txt --target-ratio 0.25 --pretty
```

Choose the compression method:

```bash
semantic-query-compressor --file example.txt --compression-method amr --pretty
semantic-query-compressor --file example.txt --compression-method llm --pretty
```

`target-ratio` is the approximate fraction of sentences to keep. A smaller value creates a shorter compressed query, but may drop important intent.

## Demo Usage

The repository includes `main.py` and `example.txt` for a quick demonstration:

```bash
python main.py
```

Choose AMR or LLM compression in the demo:

```bash
python main.py --compression-method amr
python main.py --compression-method llm
```

The demo prints the original query, compressed query, original LLM response, compressed LLM response, and these comparison metrics:

- `Retention ratio`: compressed word count divided by original word count.
- `Size reduction`: percentage of words removed from the original query.
- `Critical sentence coverage`: fraction of detected task and constraint sentences preserved in the compressed query.
- `Query similarity`: TF-IDF cosine similarity between the original query and compressed query.
- `LLM response similarity`: TF-IDF cosine similarity between the LLM response for the original query and the LLM response for the compressed query.

`main.py` requires Groq to be configured because both compression evaluation modes compare real Groq LLM responses. Add `GROQ_API_KEY` to `.env` before running it.

## Output Format

The CLI prints JSON:

```json
{
  "compressed_query": "Shorter query...",
  "original_output": "LLM response to original query...",
  "compressed_output": "LLM response to compressed query...",
  "similarity_score": 0.84,
  "compression_method": "amr"
}
```

## Running Tests

The tests do not call Groq. They use a fake LLM client so they can run offline.

```bash
pytest
```

## Important Notes

- The input `query` must be a string with at least 100 words.
- The implementation uses AMR-inspired extractive graph compression, so `compressed_query` is made from sentences already present in the original query.
- LLM-based compression may rewrite the query instead of preserving exact original sentences.
- Dense prompts with many separate requirements may compress less aggressively because task and constraint sentences are preserved intentionally.
- This is not full formal AMR parsing. Full AMR parsing requires an external parser/model, which can be added later.
- `temperature` is set to `0.0` in the Groq client to reduce response variance.
- TF-IDF similarity is useful as a baseline, but it is not perfect. Two answers can be semantically similar while using different vocabulary, which may lower the score.

## Where to Improve Next

Good next upgrades are:

1. Add a full AMR parser backend and keep the current graph compressor as a fallback.
2. Add embedding-based similarity for stronger semantic comparison.
3. Add a Groq judge prompt that classifies whether intent is preserved.
4. Store runs in a CSV or database for experiment tracking.
