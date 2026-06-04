"""
FastAPI 后端 —— RAG 文档管理接口

启动方式：
    python -m uvicorn src.api:app --host 0.0.0.0 --port 9000
"""
import os
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

try:
    from .rag import RAG
except ImportError:
    from rag import RAG

app = FastAPI(title="RAG Doc Manager")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAG()

TEMP_DIR = Path("temp_docs")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class AddDocRequest(BaseModel):
    filename: str
    content: str   # Markdown 文本


class DeleteDocRequest(BaseModel):
    filename: str


class DocDetailResponse(BaseModel):
    filename: str
    chunk_count: int
    content: str


# 接口

@app.get("/api/docs/list")
def list_docs():
    """获取所有已入库文档名称列表。"""
    try:
        docs = rag.list_documents()
        return {"documents": docs}
    except Exception as e:
        print("=" * 50)
        print("list_docs 错误:")
        traceback.print_exc()
        print("=" * 50)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/docs/add")
def add_or_update_doc(req: AddDocRequest):
    """添加新文档或更新已有文档。

    前端读取文件内容后，将 filename 和 Markdown content 传给此接口。
    若文档已存在则先删除旧版本再入库，实现覆盖更新。
    """
    print(f"收到添加文档请求: {req.filename}")
    print(f"内容长度: {len(req.content)} 字符")

    if not req.filename:
        raise HTTPException(status_code=400, detail="filename 不能为空")
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="文档内容不能为空")

    safe_name = Path(req.filename).name or req.filename
    tmp_path = TEMP_DIR / safe_name

    try:
        tmp_path.write_text(req.content, encoding="utf-8")
        print(f"临时文件已写入: {tmp_path}")
    except Exception as e:
        print(f"写入临时文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"写入临时文件失败: {e}")

    try:
        # 若同名文档已存在，先删除（实现覆盖更新）
        existing = rag.list_documents()
        print(f"当前已有文档: {existing}")
        if safe_name in existing:
            print(f"删除旧文档: {safe_name}")
            rag.delete_document(safe_name)

        print(f"开始入库: {tmp_path}")
        rag.store_func(str(tmp_path))
        print(f"入库成功: {safe_name}")
        return {"status": "ok", "filename": safe_name}
    except Exception as e:
        print("=" * 50)
        print("添加文档时发生错误:")
        traceback.print_exc()
        print("=" * 50)
        raise HTTPException(status_code=500, detail=f"入库失败: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
            print(f"临时文件已删除: {tmp_path}")


@app.post("/api/docs/delete")
def delete_doc(req: DeleteDocRequest):
    """删除指定文档。"""
    if not req.filename:
        raise HTTPException(status_code=400, detail="filename 不能为空")
    try:
        result = rag.delete_document(req.filename)
        return {"status": "ok", "message": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print("=" * 50)
        print("删除文档时发生错误:")
        traceback.print_exc()
        print("=" * 50)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/docs/detail")
def doc_detail(filename: str):
    """获取文档完整内容（用于编辑回显）。"""
    if not filename:
        raise HTTPException(status_code=400, detail="filename 不能为空")
    try:
        detail = rag.get_document_detail(filename)
        return detail
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print("=" * 50)
        print("获取文档详情时发生错误:")
        traceback.print_exc()
        print("=" * 50)
        raise HTTPException(status_code=500, detail=str(e))


# 添加 search 接口（供前端调用）
class SearchRequest(BaseModel):
    query: str


@app.post("/api/docs/search")
def search_docs(req: SearchRequest):
    """检索文档（与 8888 端口的 /chat 功能类似）"""
    try:
        results = rag.search(req.query)
        # 格式化返回结果
        formatted_results = []
        for r in results:
            for ctx in r.get("context", []):
                if ctx.get("content"):
                    formatted_results.append(ctx["content"])
        return {"results": formatted_results[:10]}  # 最多返回10条
    except Exception as e:
        print("=" * 50)
        print("检索文档时发生错误:")
        traceback.print_exc()
        print("=" * 50)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)