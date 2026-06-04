from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import requests
import os

app = FastAPI(title="CNC Agent Service")

# ====================== 配置 ======================
# TODO: 后端 API 地址（问后端同事确认）
BACKEND_API_URL = "http://47.108.254.241:8000"  # 端口问后端

# RAG 知识库地址（问后端 RAG 服务部署在哪）
RAG_API_URL = "http://127.0.0.1:9000/api/docs/search"

# 大模型配置（先用模拟，后续接入真实）
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ====================== 请求/响应模型 ======================
class AgentRequest(BaseModel):
    question: str
    tool_id: Optional[str] = None
    context: Optional[str] = None

class AgentResponse(BaseModel):
    answer: str
    figure: Optional[dict] = None

# ====================== 工具函数 ======================
def get_tool_status(tool_id: str) -> str:
    """调用后端接口获取刀具状态"""
    try:
        resp = requests.get(f"{BACKEND_API_URL}/api/tools/{tool_id}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return f"刀具{tool_id}，健康评分{data.get('health_score')}，剩余寿命{data.get('rul')}天"
        return f"未找到刀具 {tool_id}"
    except Exception as e:
        return f"查询失败: {str(e)}"

def search_knowledge(query: str) -> str:
    """调用 RAG 服务检索知识库"""
    try:
        resp = requests.get(RAG_API_URL, params={"q": query}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                return "\n---\n".join([r.get("content", "")[:300] for r in results[:3]])
        return "未找到相关知识"
    except Exception as e:
        return f"知识库查询失败: {str(e)}"

def call_llm(prompt: str) -> str:
    """调用大模型生成回答（先用模拟）"""
    # TODO: 替换成真实的大模型调用
    return f"【模拟回答】{prompt[:200]}..."

# ====================== 核心接口 ======================
@app.post("/agent", response_model=AgentResponse)
async def agent(request: AgentRequest):
    # 1. 如果有 tool_id，获取刀具状态
    tool_info = ""
    if request.tool_id:
        tool_info = get_tool_status(request.tool_id)
    
    # 2. 如果没有提供 context，自己调用 RAG 检索
    context = request.context
    if not context:
        context = search_knowledge(request.question)
    
    # 3. 构建 prompt
    prompt = f"""用户问题：{request.question}

刀具信息：{tool_info}

知识库内容：{context}

请根据以上信息，给出专业、简洁的回答。"""
    
    # 4. 调用大模型
    answer = call_llm(prompt)
    
    return AgentResponse(answer=answer, figure=None)

# ====================== 健康检查 ======================
@app.get("/health")
async def health():
    return {"status": "healthy"}

# ====================== 启动 ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)