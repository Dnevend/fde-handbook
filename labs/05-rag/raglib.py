"""第五部分实验的共用工具：语料、切分、向量、检索、生成、指标。

为什么集中在这里：块长、BM25 参数、RRF 的 k、HNSW 参数如果各脚本各写一套，
各节的数字就不可比。参数一律读 config.yaml，指标一律用本文件的定义。

只依赖标准库 + numpy + jieba + hnswlib + openai。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BENCH = ROOT / "benchmarks" / "rag"
CACHE = HERE / ".cache"
CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))

MODEL_DIR = Path(os.environ.get("FDE_MODEL_DIR", CONFIG["models"]["dir"])).expanduser()
EMBEDDING_MODEL = MODEL_DIR / CONFIG["models"]["embedding"]
RERANKER_MODEL = MODEL_DIR / CONFIG["models"]["reranker"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 语料

class _TextExtractor(HTMLParser):
    """够用的 HTML 取正文：丢掉脚本样式，块级标签换行。"""

    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}
    SKIP = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\u3000]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def strip_excerpt_header(text: str) -> str:
    return re.sub(r"^<!--[\s\S]*?-->\s*", "", text)


def pdf_pages(path: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(path) as doc:
        return [page.get_text("text") for page in doc]


def _source_text(folder: Path) -> tuple[str, list[str] | None]:
    """按语料档位取文本：excerpt 档只读节选，full 档读全文。"""
    excerpt = folder / "__excerpt.md"
    if excerpt.exists():
        return strip_excerpt_header(excerpt.read_text(encoding="utf-8")), None

    full = folder / "full.txt"
    if full.exists():
        return full.read_text(encoding="utf-8", errors="ignore"), None

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() == ".pdf":
            pages = pdf_pages(path)
            return "\n".join(pages), pages
        if path.suffix.lower() in {".html", ".htm"}:
            return html_to_text(path.read_text(encoding="utf-8", errors="ignore")), None
        if path.suffix.lower() in {".md", ".txt"}:
            return path.read_text(encoding="utf-8", errors="ignore"), None
    raise FileNotFoundError(f"{folder} 里找不到可用文本")


def authority_map() -> dict[str, str]:
    """效力层级只在 corpus/index.py 里定义一次，这里引用而不是复制。"""
    corpus_dir = str(ROOT / "corpus")
    if corpus_dir not in sys.path:
        sys.path.insert(0, corpus_dir)
    from index import AUTHORITY  # noqa: PLC0415

    return dict(AUTHORITY)


def load_docs(include_synthetic: bool | None = None) -> list[dict]:
    """返回全部语料文档。text 是正文，pages 只在 PDF 时有值。"""
    if include_synthetic is None:
        include_synthetic = CONFIG["corpus"]["synthetic"]

    manifest = yaml.safe_load((ROOT / "corpus/sources.yaml").read_text(encoding="utf-8"))["sources"]
    authority = authority_map()
    docs: list[dict] = []

    for item in manifest:
        folder = ROOT / "corpus/raw" / item["id"]
        if not folder.is_dir():
            continue
        try:
            text, pages = _source_text(folder)
        except FileNotFoundError:
            log(f"跳过 {item['id']}：本地没有原文（见 corpus/ACQUIRE.md）")
            continue
        if len(text.strip()) < 200:
            log(f"跳过 {item['id']}：抽出 {len(text)} 字，疑为扫描件")
            continue
        docs.append(
            {
                "doc_id": item["id"],
                "title": item["title"],
                "authority": authority.get(item["id"], "未分类"),
                "doc_type": item["doc_type"],
                "synthetic": False,
                "text": text,
                "pages": pages,
            }
        )

    if include_synthetic:
        for path in sorted((ROOT / "corpus/raw/synthetic").glob("*/*.md")):
            raw = path.read_text(encoding="utf-8")
            meta_match = re.match(r"^---\n([\s\S]*?)\n---\n", raw)
            meta = yaml.safe_load(meta_match.group(1)) if meta_match else {}
            body = raw[meta_match.end():] if meta_match else raw
            docs.append(
                {
                    "doc_id": str(meta.get("doc_id") or path.stem),
                    "title": path.stem,
                    "authority": str(meta.get("level") or "内部"),
                    "doc_type": meta.get("doc_kind") or "synthetic",
                    "synthetic": True,
                    "text": body,
                    "pages": None,
                }
            )

    return docs


# ---------------------------------------------------------------- 切分

CLAUSE = re.compile(r"(?m)^\s*(第[一二三四五六七八九十百零〇]+条)")
HEADING = re.compile(r"(?m)^(#{1,4})\s+(.+)$")


def _split_long(text: str, size: int) -> list[str]:
    """按句末标点把过长的一段切开，尽量不破坏句子。"""
    if len(text) <= size:
        return [text]
    sentences = re.split(r"(?<=[。！？；])", text)
    out, buf = [], ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(buf) + len(sentence) > size and buf:
            out.append(buf)
            buf = sentence
        else:
            buf += sentence
    if buf:
        out.append(buf)
    # 没有标点的超长串，硬切
    final: list[str] = []
    for piece in out:
        while len(piece) > size * 1.5:
            final.append(piece[:size])
            piece = piece[size:]
        if piece:
            final.append(piece)
    return final


def chunk_structure(doc: dict) -> list[dict]:
    """先按法规条款/标题切，再把过长的块按句子切开；不跨条款合并。"""
    size = CONFIG["corpus"]["chunk_size"]
    text = doc["text"]
    chunks: list[dict] = []

    parts: list[tuple[str | None, str]] = []
    matches = list(CLAUSE.finditer(text))
    if len(matches) >= 3:
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            parts.append((match.group(1), text[match.start():end].strip()))
    else:
        headings = list(HEADING.finditer(text))
        if headings:
            for i, match in enumerate(headings):
                end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
                parts.append((match.group(2).strip(), text[match.start():end].strip()))
        else:
            parts = [(None, para.strip()) for para in re.split(r"\n\s*\n", text) if para.strip()]

    for label, body in parts:
        if not body:
            continue
        pieces = _split_long(body, size)
        for piece in pieces:
            if len(piece.strip()) < CONFIG["corpus"]["min_chunk"]:
                continue
            chunks.append(
                {
                    "doc_id": doc["doc_id"],
                    "clause_no": label if label and label.startswith("第") else None,
                    "section": label if label and not label.startswith("第") else None,
                    "text": piece.strip(),
                }
            )
    return chunks


def chunk_fixed(doc: dict) -> list[dict]:
    """按字符窗口切，窗口边界尽量落在句末。"""
    size = CONFIG["corpus"]["chunk_size"]
    overlap = CONFIG["corpus"]["chunk_overlap"]
    text = re.sub(r"\s+", " ", doc["text"]).strip()
    out: list[dict] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("。"), window.rfind("！"), window.rfind("？"))
            if cut > size * 0.6:
                end = start + cut + 1
        piece = text[start:end].strip()
        if len(piece) >= CONFIG["corpus"]["min_chunk"]:
            out.append({"doc_id": doc["doc_id"], "clause_no": None, "section": None, "text": piece})
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def build_chunks(strategy: str = "structure", include_synthetic: bool | None = None) -> list[dict]:
    docs = load_docs(include_synthetic)
    chunks: list[dict] = []
    for doc in docs:
        pieces = chunk_structure(doc) if strategy == "structure" else chunk_fixed(doc)
        for piece in pieces:
            piece["chunk_id"] = f"{doc['doc_id']}#{len(chunks):05d}"
            piece["authority"] = doc["authority"]
            piece["doc_type"] = doc["doc_type"]
            piece["synthetic"] = doc["synthetic"]
            piece["title"] = doc["title"]
            piece["char_len"] = len(piece["text"])
            chunks.append(piece)
    return chunks


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cached_chunks(strategy: str = "structure", include_synthetic: bool | None = None) -> list[dict]:
    """缓存文件名区分「全部语料」与「仅真实语料」。

    质量实验（5.3/5.4/5.5/5.6/5.7/5.8）只含真实文档；5.9 的权限与增量实验需要合成件。
    """
    suffix = "" if include_synthetic is None else ("-real" if not include_synthetic else "-all")
    path = CACHE / f"chunks-{strategy}{suffix}.jsonl"
    if not path.exists():
        rows = build_chunks(strategy, include_synthetic)
        save_jsonl(path, rows)
        log(f"切分 {strategy}{suffix}：{len(rows)} 块 → {path.name}")
    return load_jsonl(path)


# ---------------------------------------------------------------- 服务

def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait(port: int, seconds: int = 120) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if _healthy(port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"端口 {port} 的服务没起来，看 {CACHE}/logs/")


def ensure_embedding_server() -> int:
    port = CONFIG["servers"]["embedding_port"]
    if _healthy(port):
        return port
    if not EMBEDDING_MODEL.exists():
        raise RuntimeError(f"缺 {EMBEDDING_MODEL}，先跑 labs/05-rag/download-models.sh")
    (CACHE / "logs").mkdir(parents=True, exist_ok=True)
    logf = (CACHE / "logs" / "embedding.log").open("a")
    subprocess.Popen(
        [
            "llama-server", "-m", str(EMBEDDING_MODEL),
            "--host", "127.0.0.1", "--port", str(port),
            "--embeddings", "--pooling", "last",
            "-c", "4096", "-b", "2048", "-ub", "2048", "-ngl", "99",
            "--alias", "qwen3-embedding",
        ],
        stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _wait(port)
    return port


def ensure_reranker_server() -> int:
    port = CONFIG["servers"]["reranker_port"]
    if _healthy(port):
        return port
    if not RERANKER_MODEL.exists():
        raise RuntimeError(f"缺 {RERANKER_MODEL}，先跑 labs/05-rag/download-models.sh")
    (CACHE / "logs").mkdir(parents=True, exist_ok=True)
    logf = (CACHE / "logs" / "reranker.log").open("a")
    subprocess.Popen(
        [
            "llama-server", "-m", str(RERANKER_MODEL),
            "--host", "127.0.0.1", "--port", str(port),
            "--reranking", "-c", "2048", "-b", "2048", "-ub", "2048", "-ngl", "99",
            "--alias", "bge-reranker",
        ],
        stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _wait(port)
    return port


def ensure_llm_server(ctx: int = 8192, port: int | None = None) -> int:
    port = port or CONFIG["servers"]["llm_port"]
    if _healthy(port):
        return port
    env_sh = ROOT / "labs" / "env.sh"
    result = subprocess.run(
        ["bash", "-c", f'source "{env_sh}" >/dev/null && printf %s "$MODEL"'],
        capture_output=True, text=True,
    )
    model = result.stdout.strip()
    if not model:
        raise RuntimeError("找不到 Qwen3.5-9B-Q4_K_M.gguf，先 source labs/env.sh（见 0.3 节）")
    (CACHE / "logs").mkdir(parents=True, exist_ok=True)
    logf = (CACHE / "logs" / f"llm-{port}.log").open("a")
    subprocess.Popen(
        [
            "llama-server", "-m", model,
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(ctx), "-ngl", "99", "--jinja",
            "--alias", "qwen3.5-9b",
        ],
        stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _wait(port, seconds=180)
    return port


def stop_servers() -> None:
    subprocess.run(["pkill", "-f", "llama-server"], check=False)


# ---------------------------------------------------------------- 向量与重排

def _post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read())


def embed_texts(texts: list[str], kind: str = "doc") -> np.ndarray:
    """调本地 embedding 端点。kind=query 时加 Qwen3 的指令前缀。"""
    if not texts:
        return np.zeros((0, CONFIG["embedding"]["dim_full"]), dtype=np.float32)
    prefix = ""
    if kind == "query":
        prefix = f"Instruct: {CONFIG['embedding']['query_instruction']}\nQuery: "
    payload_texts = [prefix + text for text in texts]
    port = ensure_embedding_server()
    vectors: list[list[float]] = []
    batch = CONFIG["embedding"]["batch"]
    for start in range(0, len(payload_texts), batch):
        chunk = payload_texts[start:start + batch]
        resp = _post_json(
            f"http://127.0.0.1:{port}/v1/embeddings",
            {"model": "qwen3-embedding", "input": chunk},
        )
        ordered = sorted(resp["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in ordered)
        if (start // batch) % 20 == 0:
            log(f"embedding {start + len(chunk)}/{len(payload_texts)}")
    array = np.asarray(vectors, dtype=np.float32)
    return l2_normalize(array)


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def matryoshka(array: np.ndarray, dim: int) -> np.ndarray:
    """MRL：截断前 dim 维后重新归一化（Qwen3-Embedding 支持）。"""
    return l2_normalize(array[:, :dim].copy())


def quantize_rows(array: np.ndarray, bits: int = 8) -> np.ndarray:
    """逐行对称量化再反量化，模拟把向量按 int8 存储。"""
    max_abs = np.abs(array).max(axis=1, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    levels = 2 ** (bits - 1) - 1
    scale = max_abs / levels
    quantized = np.clip(np.round(array / scale), -levels - 1, levels)
    return l2_normalize((quantized * scale).astype(np.float32))


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    port = ensure_reranker_server()
    payload = {
        "model": "bge-reranker",
        "query": query,
        "documents": documents,
        "top_n": top_n or len(documents),
    }
    resp = _post_json(f"http://127.0.0.1:{port}/v1/rerank", payload, timeout=600)
    return resp["results"]


def count_tokens(text: str, port: int | None = None) -> int:
    """用服务的分词器数 token，避免拿字符数估算上下文长度。"""
    port = port or ensure_llm_server()
    resp = _post_json(f"http://127.0.0.1:{port}/tokenize", {"content": text})
    return len(resp.get("tokens", []))


def chat_stream(messages: list[dict], max_tokens: int | None = None, port: int | None = None) -> tuple[str, float]:
    """流式调用，返回（完整回答, 首 token 延迟秒）。"""
    from openai import OpenAI

    port = port or ensure_llm_server()
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="sk-local")
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model="qwen3.5-9b",
        messages=messages,
        temperature=CONFIG["generation"]["temperature"],
        top_p=CONFIG["generation"]["top_p"],
        max_tokens=max_tokens or CONFIG["generation"]["max_tokens"],
        seed=CONFIG["generation"]["seed"],
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    first = None
    parts: list[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta and first is None:
            first = time.perf_counter()
        parts.append(delta)
    return "".join(parts), (first - start) if first else float("nan")


def chat(messages: list[dict], max_tokens: int | None = None, port: int | None = None) -> str:
    from openai import OpenAI

    port = port or ensure_llm_server()
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="sk-local")
    response = client.chat.completions.create(
        model="qwen3.5-9b",
        messages=messages,
        temperature=CONFIG["generation"]["temperature"],
        top_p=CONFIG["generation"]["top_p"],
        max_tokens=max_tokens or CONFIG["generation"]["max_tokens"],
        seed=CONFIG["generation"]["seed"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------- 指标

def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def chunk_is_relevant(chunk: dict, gold: dict) -> bool:
    """一条 gold 出处对应哪些块算命中：doc 必须对；写了条款号就还要求条款号对。"""
    if chunk["doc_id"] != gold["doc"]:
        return False
    clause = gold.get("clause")
    if clause and chunk.get("clause_no"):
        return clause == chunk["clause_no"]
    return True


def relevant_ids(chunks: list[dict], golds: list[dict]) -> set[str]:
    return {chunk["chunk_id"] for chunk in chunks if any(chunk_is_relevant(chunk, gold) for gold in golds)}


# ---------------------------------------------------------------- 落盘

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
