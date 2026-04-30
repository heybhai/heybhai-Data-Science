"""
A tiny RAG pipeline for learning.

What it does:
1. Loads text from C:\\Users\\Harsheet Gandhi\\Documents\\content.txt
2. Splits it into overlapping chunks
3. Builds a simple TF-IDF index in memory
4. Retrieves the most relevant chunks for your question
5. Sends those chunks to Gemini Flash to answer your question

Before running:
    pip install google-genai

Then set your API key:
    PowerShell:  $env:GEMINI_API_KEY="your_api_key_here"
    Command Prompt: set GEMINI_API_KEY=your_api_key_here
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONTENT_PATH = Path(r"C:\Users\Harsheet Gandhi\Documents\content.txt")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class Chunk:
    id: int
    text: str
    start_word: int
    end_word: int


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer used by both indexing and querying."""
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def load_document(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Check the path or pass --file with another text file."
        )

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"{path} is empty. Add some text first, then run again.")

    return text


def split_into_chunks(text: str, chunk_words: int = 120, overlap_words: int = 30) -> list[Chunk]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero.")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be at least 0 and smaller than chunk_words.")

    words = text.split()
    chunks: list[Chunk] = []
    step = chunk_words - overlap_words

    for chunk_id, start in enumerate(range(0, len(words), step), start=1):
        end = min(start + chunk_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(id=chunk_id, text=chunk_text, start_word=start, end_word=end))
        if end == len(words):
            break

    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_vectors: list[dict[str, float]] = []
        self.idf: dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        document_frequency: defaultdict[str, int] = defaultdict(int)
        tokenized_chunks = [tokenize(chunk.text) for chunk in self.chunks]

        for tokens in tokenized_chunks:
            for token in set(tokens):
                document_frequency[token] += 1

        total_docs = len(self.chunks)
        self.idf = {
            token: math.log((1 + total_docs) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

        self.doc_vectors = [self._vectorize(tokens) for tokens in tokenized_chunks]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}

        counts = Counter(tokens)
        token_count = len(tokens)
        return {
            token: (count / token_count) * self.idf.get(token, 0.0)
            for token, count in counts.items()
        }

    @staticmethod
    def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0

        shared_tokens = set(left) & set(right)
        dot_product = sum(left[token] * right[token] for token in shared_tokens)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot_product / (left_norm * right_norm)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        query_vector = self._vectorize(tokenize(question))
        scored_results = [
            SearchResult(chunk=chunk, score=self._cosine_similarity(query_vector, doc_vector))
            for chunk, doc_vector in zip(self.chunks, self.doc_vectors)
        ]

        scored_results.sort(key=lambda result: result.score, reverse=True)
        return scored_results[:top_k]


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        f"[Chunk {result.chunk.id} | score={result.score:.3f}]\n{result.chunk.text}"
        for result in results
    )
    return (
        "Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_gemini_answer(question: str, results: list[SearchResult], model: str) -> str:
    """Generate the final answer with Gemini, using retrieved chunks as context."""
    if not results:
        return "I could not find relevant context in the document for that question."

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "The Gemini Python package is not installed. Run: pip install google-genai"
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. In PowerShell, run: "
            '$env:GEMINI_API_KEY="AIzaSyCz51I6_oedJlHzrOW1rc_-49SWIeor67I"'
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(question, results),
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a helpful RAG assistant. Answer using only the provided context. "
                "If the context does not contain the answer, say that the document does not say."
            ),
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    return response.text.strip()


def ask_question(
    retriever: TfidfRetriever,
    question: str,
    top_k: int,
    model: str,
    show_prompt: bool,
) -> None:
    results = retriever.search(question, top_k=top_k)

    print("\nRetrieved context")
    print("-" * 60)
    if not results:
        print("No matching chunks found.")
    for result in results:
        print(f"Chunk {result.chunk.id} | score={result.score:.3f}")
        print(result.chunk.text)
        print("-" * 60)

    print("\nGemini answer")
    print("-" * 60)
    try:
        print(generate_gemini_answer(question, results, model=model))
    except RuntimeError as error:
        print(error)

    if show_prompt:
        print("\nPrompt sent to the LLM")
        print("-" * 60)
        print(build_prompt(question, results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning RAG pipeline over a local text file.")
    parser.add_argument("--file", type=Path, default=DEFAULT_CONTENT_PATH, help="Path to the text file.")
    parser.add_argument("--chunk-words", type=int, default=120, help="Words per chunk.")
    parser.add_argument("--overlap-words", type=int, default=30, help="Words repeated between chunks.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model to use.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the full prompt sent to Gemini.")
    parser.add_argument("--question", type=str, help="Ask one question and exit.")
    args = parser.parse_args()

    document = load_document(args.file)
    chunks = split_into_chunks(
        document,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    retriever = TfidfRetriever(chunks)

    print(f"Loaded: {args.file}")
    print(f"Chunks: {len(chunks)}")
    print(f"Model: {args.model}")

    if args.question:
        ask_question(
            retriever,
            args.question,
            top_k=args.top_k,
            model=args.model,
            show_prompt=args.show_prompt,
        )
        return

    print("\nAsk questions about the document. Press Enter without typing to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            print("Goodbye.")
            break
        ask_question(
            retriever,
            question,
            top_k=args.top_k,
            model=args.model,
            show_prompt=args.show_prompt,
        )


if __name__ == "__main__":
    main()
