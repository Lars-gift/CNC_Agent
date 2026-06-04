import os
import json
import sqlite3
import uuid
import re
import logging
from pathlib import Path
from typing import List

import numpy as np
import chromadb
import jieba
from rank_bm25 import BM25Okapi

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
except ImportError:
    UnstructuredMarkdownLoader = None

# 声明模块专用logger（使用全局配置）
logger = logging.getLogger(__name__)


class RAG:

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        embedding_model: str = "text-embedding-v4"
    ):
        self.host = host
        self.port = port
        self.embedding_model = embedding_model
        self.collection_name = "cut_rag"
        self.token_name = "token_bm25.db"

        self.embedding = self._init_embedding()
        self.chroma_client = chromadb.HttpClient(host=self.host, port=self.port)
        
        if UnstructuredMarkdownLoader is not None:
            _md_loader = lambda p: UnstructuredMarkdownLoader(str(p))
        else:
            _md_loader = lambda p: TextLoader(str(p), encoding="utf-8")

        self.loader_mapping = {
            ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
            ".pdf": lambda p: UnstructuredPDFLoader(str(p), mode="elements"),
            ".docx": lambda p: UnstructuredWordDocumentLoader(str(p), mode="elements"),
            ".md": lambda p: TextLoader(str(p), encoding="utf-8"),
        }
        
        self.fine_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=0,
            separators=["；", "，", " ", ""],
            length_function=len
        )
        
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embedding
        )

        self._doc_ids: List[str] = []
        self._tokenized_docs: List[List[str]] = []
        self._bm25 = None

        try:
            conn = sqlite3.connect(self.token_name)
            cursor = conn.cursor()
            cursor.execute("SELECT doc_id, tokens FROM bm25_tokens")
            rows = cursor.fetchall()
            conn.close()
            if rows:
                self._doc_ids = [row[0] for row in rows]
                self._tokenized_docs = [json.loads(r[1]) for r in rows]
                self._bm25 = BM25Okapi(self._tokenized_docs)
        except sqlite3.OperationalError:
            pass
            
    def _get_collection(self):
        return self.vector_store._collection

    def _refresh_bm25(self):
        if self._doc_ids and self._tokenized_docs:
            self._bm25 = BM25Okapi(self._tokenized_docs)
        else:
            self._bm25 = None

    def _load_document_paths(self) -> List[str]:
        conn = sqlite3.connect(self.token_name)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT file_path FROM bm25_tokens")
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []
        conn.close()
        return [row[0] for row in rows if row[0]]

    def list_documents(self) -> List[str]:
        """返回当前数据库中已入库的文档文件名列表。"""
        file_paths = self._load_document_paths()
        seen = set()
        names = []
        for path in file_paths:
            name = Path(path).name
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names

    def add_document(self, docs_path: str):
        """往数据库中新增文档（与 store_func 等价）。"""
        return self.store_func(docs_path)

    def delete_document(self, doc_name: str):
        """删除指定名称的文档及其向量、BM25 记录。"""
        file_paths = self._load_document_paths()
        if not file_paths:
            raise ValueError("当前数据库中没有任何文档可删除")

        exact_matches = [fp for fp in file_paths if fp == doc_name]
        if exact_matches:
            target_paths = exact_matches
        else:
            candidates = [fp for fp in file_paths if Path(fp).name == doc_name]
            if not candidates:
                raise ValueError(f"未找到名称为 {doc_name} 的文档")
            if len(candidates) > 1:
                raise ValueError(
                    f"存在多个同名文档，请提供更精确路径：{candidates}"
                )
            target_paths = candidates

        target_path = target_paths[0]
        collection = self._get_collection()
        file_data = collection.get(
            where={"file_path": {"$eq": target_path}},
            include=["documents"]
        )
        ids = file_data.get("ids", [])
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
            ids = ids[0]

        # 删除向量库中的记录
        try:
            collection.delete(ids=ids)
        except Exception:
            collection.delete(where={"file_path": {"$eq": target_path}})

        # 删除 BM25 SQLite 记录
        conn = sqlite3.connect(self.token_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bm25_tokens WHERE file_path = ?", (target_path,))
        conn.commit()
        conn.close()

        # 更新内存索引
        deleted_ids = set(ids)
        original_doc_ids = self._doc_ids
        original_tokens = self._tokenized_docs
        self._doc_ids = [doc_id for doc_id in original_doc_ids if doc_id not in deleted_ids]
        self._tokenized_docs = [tokens for doc_id, tokens in zip(original_doc_ids, original_tokens) if doc_id not in deleted_ids]
        self._refresh_bm25()

        return f"成功删除文档：{Path(target_path).name}"

    def get_document_detail(self, doc_name: str) -> dict:
        """获取指定文档的完整内容（用于编辑回显）。"""
        file_paths = self._load_document_paths()
        candidates = [fp for fp in file_paths if Path(fp).name == doc_name]
        if not candidates:
            raise ValueError(f"未找到名称为 {doc_name} 的文档")
        target_path = candidates[0]

        collection = self._get_collection()
        file_data = collection.get(
            where={"file_path": {"$eq": target_path}},
            include=["documents", "metadatas"]
        )
        chunks = list(zip(file_data["ids"], file_data["documents"], file_data["metadatas"]))
        chunks.sort(key=lambda x: x[2].get("chunk_index", 0))

        content = "\n\n".join(doc for _, doc, _ in chunks)
        return {
            "filename": Path(target_path).name,
            "file_path": target_path,
            "chunk_count": len(chunks),
            "content": content,
        }

    def _init_embedding(self) -> DashScopeEmbeddings:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("阿里云API Key 未设置，请设置环境变量 DASHSCOPE_API_KEY")
        return DashScopeEmbeddings(
            model=self.embedding_model,
            dashscope_api_key=api_key
        )
    
    def store_func(self, docs_path: str):
        """入库主路"""
        file_path = Path(docs_path)
        if not file_path.exists():
            raise ValueError(f"文件 {file_path} 不存在")
        ext = file_path.suffix.lower()

        loader_factory = self.loader_mapping.get(ext)
        if not loader_factory:
            raise ValueError(f"不支持的文件类型: {ext}")
        
        # PDF 图片预提取到本地
        image_dir = Path("images") / file_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        page_image_map = {}
        
        if ext == ".pdf":
            import fitz
            pdf_doc = fitz.open(file_path)
            for page_idx in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_idx)
                img_list = page.get_images(full=True)
                page_num = page_idx + 1
                paths = []
                for img_idx, img in enumerate(img_list, start=1):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_name = f"{file_path.stem}_page{page_num}_img{img_idx}.png"
                    img_path = image_dir / img_name
                    pix.save(str(img_path))
                    pix = None
                    paths.append(str(img_path))
                if paths:
                    page_image_map[page_num] = paths
            pdf_doc.close()
        
        loader = loader_factory(file_path)
        raw_docs = loader.load()
        
        # 元素级后处理
        processed = []
        page_img_cursor = {}
        
        for doc in raw_docs:
            meta = doc.metadata
            cat = meta.get("category") or meta.get("element_type") or ""
            page_num = meta.get("page_number")
            
            # 表格 → Markdown 表格
            if cat in ("Table", "table"):
                lines = [l.strip() for l in doc.page_content.splitlines() if l.strip()]
                rows = []
                for line in lines:
                    cols = [c.strip() for c in re.split(r'\t| {2,}', line) if c.strip()]
                    if cols:
                        rows.append(cols)
                if len(rows) >= 2:
                    max_cols = max(len(r) for r in rows)
                    md = []
                    for j, r in enumerate(rows):
                        r += [''] * (max_cols - len(r))
                        md.append('| ' + ' | '.join(r) + ' |')
                        if j == 0:
                            md.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
                    doc.page_content = '\n'.join(md)
                processed.append(doc)
            
            # 图片 → 只保留有本地真实路径的，无路径直接丢弃
            elif cat in ("Image", "image", "Picture"):
                alt = meta.get("alt_text", "") or meta.get("text_as_html", "") or "文档图片"
                alt = alt[:50].replace('\n', ' ')
                
                local_img = ""
                if ext == ".pdf" and page_num and page_num in page_image_map:
                    idx = page_img_cursor.get(page_num, 0)
                    if idx < len(page_image_map[page_num]):
                        local_img = page_image_map[page_num][idx]
                        page_img_cursor[page_num] = idx + 1
                
                if local_img:
                    doc.page_content = f"![{alt}]({local_img})"
                    processed.append(doc)
            
            else:
                processed.append(doc)
        
        # 二次切分：表格/图片保持独立；正文按"。"分句，短句独立，长句内部细切
        text_docs = []
        table_image_docs = []
        for doc in processed:
            cat = doc.metadata.get("category") or doc.metadata.get("element_type") or ""
            if cat in ("Table", "table", "Image", "image", "Picture"):
                table_image_docs.append(doc)
            else:
                text_docs.append(doc)

        # 正文按页合并后，严格按"。"切分
        page_texts = {}
        for doc in text_docs:
            page_num = doc.metadata.get("page_number", 0)
            page_texts.setdefault(page_num, []).append(doc)

        text_chunks = []
        for page_num in sorted(page_texts.keys()):
            merged = "\n".join([d.page_content for d in page_texts[page_num]])
            sentences = merged.split("。")
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sentence = sent + "。"
                if len(sentence) <= 256:
                    text_chunks.append(Document(
                        page_content=sentence,
                        metadata={"page_number": page_num}
                    ))
                else:
                    subs = self.fine_splitter.split_text(sentence)
                    for sub in subs:
                        text_chunks.append(Document(
                            page_content=sub,
                            metadata={"page_number": page_num}
                        ))

        chunks = table_image_docs + text_chunks
        
        # 统一编号
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                "file_path": str(file_path),
                "chunk_index": i,
                "is_table": chunk.page_content.startswith('| '),
                "is_image": chunk.page_content.startswith('!['),
            }
        
        # 向量化入库
        texts = [c.page_content for c in chunks]
        embeddings = self.embedding.embed_documents(texts)
        
        new_ids = [str(uuid.uuid4()) for _ in chunks]
        collection = self.vector_store._collection
        collection.add(
            ids=new_ids,
            documents=texts,
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings
        )

        # 切词写入 SQLite
        new_tokenized = [list(jieba.cut(c.page_content)) for c in chunks]
        conn = sqlite3.connect(self.token_name)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bm25_tokens (
                doc_id TEXT PRIMARY KEY,
                tokens TEXT NOT NULL,
                file_path TEXT,
                chunk_index INTEGER
            )
        """)
        conn.commit()

        for doc_id, tokens, chunk in zip(new_ids, new_tokenized, chunks):
            cursor.execute(
                "INSERT INTO bm25_tokens (doc_id, tokens, file_path, chunk_index) VALUES (?, ?, ?, ?)",
                (doc_id, json.dumps(tokens, ensure_ascii=False), str(file_path), chunk.metadata["chunk_index"])
            )
        conn.commit()
        conn.close()

        self._doc_ids.extend(new_ids)
        self._tokenized_docs.extend(new_tokenized)
        self._bm25 = BM25Okapi(self._tokenized_docs)
        
        return "成功入库"
        
    def search(self, query: str) -> List[dict]:
        """检索函数"""
        if self._bm25 is None or not self._doc_ids:
            return []

        collection = self.vector_store._collection
        n_docs = len(self._doc_ids)

        # BM25 检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = np.array(self._bm25.get_scores(tokenized_query), dtype=float)

        # 向量检索
        query_embedding = self.embedding.embed_query(query)
        vec_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_docs,
            include=["distances"]
        )
        vec_ids = vec_result["ids"][0]
        vec_distances = vec_result["distances"][0]

        # 按 doc_id 对齐向量分数
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self._doc_ids)}
        distances = np.zeros(n_docs, dtype=float)
        for vid, vdist in zip(vec_ids, vec_distances):
            if vid in id_to_idx:
                distances[id_to_idx[vid]] = vdist
            
        # 归一化 + 6:4 融合
        def _norm(scores, invert=False):
            scores = scores.astype(float)
            min_s, max_s = scores.min(), scores.max()
            if max_s == min_s:
                return np.ones_like(scores) * 0.5
            n = (scores - min_s) / (max_s - min_s)
            return 1.0 - n if invert else n
        
        vec_norm = _norm(distances, invert=True)
        bm25_norm = _norm(bm25_scores, invert=False)
        final_scores = 0.4 * vec_norm + 0.6 * bm25_norm

        # 取 Top 5 中心句
        top_indices = np.argsort(final_scores)[::-1][:5]
        top_ids = [self._doc_ids[i] for i in top_indices]

        top_meta_raw = collection.get(ids=top_ids, include=["metadatas"])
        top_metas = {i: m for i, m in zip(top_meta_raw["ids"], top_meta_raw["metadatas"])}

        results = []
        covered = set()  # 记录已被覆盖的 (file_path, chunk_index)，用于去重
        
        for idx in top_indices:
            doc_id = self._doc_ids[idx]
            meta = top_metas.get(doc_id, {})
            file_path = meta.get("file_path")
            chunk_index = meta.get("chunk_index")
            
            if not file_path or chunk_index is None:
                continue
            
            # 如果该中心句已被前面更高分结果的上下文覆盖，直接跳过
            if (file_path, chunk_index) in covered:
                continue

            # 回查同文件全部 chunks，按 chunk_index 排序
            file_data = collection.get(
                where={"file_path": {"$eq": file_path}},
                include=["documents", "metadatas"]
            )
            file_chunks = list(zip(file_data["ids"], file_data["documents"], file_data["metadatas"]))
            file_chunks.sort(key=lambda x: x[2].get("chunk_index", 0))

            # 定位中心句位置
            anchor_pos = 0
            for pos, (fid, _, fmeta) in enumerate(file_chunks):
                if fid == doc_id and fmeta.get("chunk_index") == chunk_index:
                    anchor_pos = pos
                    break
            
            # 前后各 5 句 + 本身，边界自动收缩
            start = max(0, anchor_pos - 5)
            end = min(len(file_chunks), anchor_pos + 6)

            context = []
            for pos in range(start, end):
                fid, fdoc, fmeta = file_chunks[pos]
                cidx = fmeta.get("chunk_index")
                # 标记该 chunk 已被覆盖，后续低分中心句若落在此范围则跳过
                covered.add((file_path, cidx))
                context.append({
                    "content": fdoc,
                    "metadata": fmeta,
                    "is_anchor": (fid == doc_id)
                })
            
            results.append({
                "final_score": round(float(final_scores[idx]), 4),
                "vector_score": round(float(vec_norm[idx]), 4),
                "bm25_score": round(float(bm25_norm[idx]), 4),
                "context": context
            })
            
        return results


# ──────────────────────────────────────────────
#  RAG 数据库增删查改测试
# ──────────────────────────────────────────────


def test_list_documents(rag: RAG):
    """查：列出所有已入库文档"""
    docs = rag.list_documents()
    print(f"[list_documents] 当前文档数: {len(docs)}")
    for d in docs:
        print(f"  - {d}")
    return docs


def test_add_document(rag: RAG, file_path: str):
    """增：入库文档"""
    print(f"[add_document] 入库: {file_path}")
    try:
        result = rag.add_document(file_path)
        print(f"  结果: {result}")
        return True
    except Exception as e:
        print(f"  失败: {e}")
        return False


def test_search(rag: RAG, query: str):
    """查：检索"""
    print(f"[search] 检索: {query}")
    results = rag.search(query)
    print(f"  返回结果数: {len(results)}")
    for i, r in enumerate(results):
        print(f"  #{i} score={r['final_score']} vec={r['vector_score']} bm25={r['bm25_score']}")
        for ctx in r["context"]:
            marker = "★" if ctx["is_anchor"] else " "
            snippet = ctx["content"][:80].replace("\n", "\\n")
            print(f"    [{marker}] {snippet}...")
    return results


def test_delete_document(rag: RAG, doc_name: str):
    """删：按名称删除文档"""
    print(f"[delete_document] 删除: {doc_name}")
    try:
        result = rag.delete_document(doc_name)
        print(f"  结果: {result}")
        return True
    except Exception as e:
        print(f"  失败: {e}")
        return False


def test_re_add_document(rag: RAG, file_path: str):
    """改：重新入库同一文档（覆盖更新）"""
    print(f"[re_add_document] 重新入库（更新）: {file_path}")
    # 先按文件名删除旧记录
    doc_name = Path(file_path).name
    try:
        rag.delete_document(doc_name)
        print(f"  已删除旧版本")
    except ValueError:
        print(f"  无旧版本，直接新增")
    result = rag.add_document(file_path)
    print(f"  结果: {result}")
    return result


if __name__ == "__main__":
    import sys

    rag = RAG()

    # 测试代码
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = r"D:\code\CUT\cut_rag\data\数控机床维修实例_宗建军.pdf"

    print("=" * 50)
    print("RAG 数据库 增删查改 测试")
    print("=" * 50)

    # 1. 查 —— 列表
    print("\n>>> 1. 查：列出当前文档")
    test_list_documents(rag)

    # 2. 增 —— 入库
    print("\n>>> 2. 增：入库文档")
    test_add_document(rag, test_file)

    # 3. 查 —— 列表确认
    print("\n>>> 3. 查：入库后列表")
    test_list_documents(rag)

    # 4. 查 —— 检索
    print("\n>>> 4. 查：检索")
    test_search(rag, "料位计控制")

    # 5. 改 —— 重新入库
    print("\n>>> 5. 改：重新入库同一文档")
    test_re_add_document(rag, test_file)

    # 6. 查 —— 再列表确认
    print("\n>>> 6. 查：更新后列表")
    test_list_documents(rag)

    # 7. 删 —— 按文件名删除
    print("\n>>> 7. 删：删除文档")
    test_delete_document(rag, Path(test_file).name)

    # 8. 查 —— 最终列表
    print("\n>>> 8. 查：删除后列表")
    test_list_documents(rag)

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)