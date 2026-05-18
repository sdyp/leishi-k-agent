

import asyncio
import json
import os
import sqlite3
import re
import logging
import time
import uuid
from typing import Dict, List, AsyncGenerator, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import Counter

import random

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.agents import create_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from openai import OpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

import pickle


# ========== 加载配置 ==========
load_dotenv()
# 环境变量
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")


# ========== 生产级配置 ==========
class Settings:
    """集中配置管理"""
    # API 配置
    deepseek_api_key: str = DEEPSEEK_API_KEY
    dashscope_api_key: str = DASHSCOPE_API_KEY

    # 向量库配置
    faiss_dir: str = "faiss_indexes"

    # 重排序配置
    rerank_timeout: int = 10
    rerank_max_retries: int = 2

    # 并发控制
    max_concurrent_requests: int = 10
    request_timeout: int = 30

    # 安全配置
    max_input_length: int = 500
    rate_limit_per_minute: int = 20

    # 数据库
    db_path: str = "db/memory.db"
    db_pool_size: int = 5

    # 日志
    log_level: str = "INFO"


settings = Settings()


# ========== 生产级日志配置 ==========
class JSONFormatter(logging.Formatter):
    """JSON 格式日志"""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        return json.dumps(log_entry, ensure_ascii=False)


# 配置根日志
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger("ls_agent")

# 控制台 Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(JSONFormatter())
logger.addHandler(console_handler)

# 文件 Handler
file_handler = logging.FileHandler("logs/ls_agent.log", encoding="utf-8")
file_handler.setFormatter(JSONFormatter())
logger.addHandler(file_handler)

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)


# ========== 请求模型（输入验证）==========
class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., min_length=1, max_length=settings.max_input_length)
    session_id: str = Field(default="default", max_length=100)

    @field_validator('message')
    @classmethod
    def sanitize_input(cls, v):
        """输入清洗，防止注入"""
        # 移除潜在危险字符
        v = re.sub(r'[<>{}`]', '', v)
        # 限制长度
        v = v.strip()
        if len(v) > settings.max_input_length:
            v = v[:settings.max_input_length]
        return v


# ========== 限流配置 ==========
limiter = Limiter(key_func=get_remote_address)

# ========== 数据库连接池 ==========
os.makedirs("db", exist_ok=True)
engine = create_engine(
    f"sqlite:///{settings.db_path}",
    poolclass=QueuePool,
    pool_size=settings.db_pool_size,
    max_overflow=10,
    pool_pre_ping=True
)

# ========== 初始化 ==========
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=settings.dashscope_api_key
)

model = init_chat_model(model="deepseek:deepseek-chat")


# ========== 加载向量库 ==========
def load_kb(name):
    save_path = f"{settings.faiss_dir}/{name}"
    if os.path.exists(save_path):
        logger.info(f"加载已有向量库: {name}")
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    logger.warning(f"向量库不存在: {name}")
    return None

# ========== 加载 BM25 检索器 ==========
device_bm25 = None
song_bm25 = None
sales_bm25 = None

device_path = "faiss_indexes/device_bm25"
song_path = "faiss_indexes/song_bm25"
sales_path = "faiss_indexes/sales_bm25"

# ========== 加载 BM25 检索器 ==========
device_bm25 = None
song_bm25 = None
sales_bm25 = None

if os.path.exists("faiss_indexes/device_bm25.pkl"):
    with open("faiss_indexes/device_bm25.pkl", "rb") as f:
        device_bm25 = pickle.load(f)
    print("设备库 BM25 加载完成")

if os.path.exists("faiss_indexes/song_bm25.pkl"):
    with open("faiss_indexes/song_bm25.pkl", "rb") as f:
        song_bm25 = pickle.load(f)
    print("歌曲库 BM25 加载完成")

if os.path.exists("faiss_indexes/sales_bm25.pkl"):
    with open("faiss_indexes/sales_bm25.pkl", "rb") as f:
        sales_bm25 = pickle.load(f)
    print("销售库 BM25 加载完成")



device_kb = load_kb("device_kb")
song_kb = load_kb("song_kb")
sales_kb = load_kb("sales_kb")

# 结果常量
NO_RESULT_DEVICE = "抱歉，我没有找到相关的设备信息。请尝试换个关键词，或联系雷石客服 400-881-6666。"
NO_RESULT_SONG = "抱歉，我没有找到相关的歌曲信息。请尝试换个关键词，或联系雷石客服 400-881-6666。"
NO_RESULT_SALE = "抱歉，我没有找到相关的销售数据。请尝试换个关键词，或联系雷石客服 400-881-6666。"

# 重排序客户端
rerank_client = OpenAI(
    api_key=settings.dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-api/v1",
    timeout=settings.rerank_timeout
)

# ========== 关键词定义 ==========
KEYWORD_MAP = {
    "device": [
        "黑屏", "蓝牙", "wifi", "连接", "安装", "配对", "升级", "麦克风",
        "声音", "机顶盒", "音响", "电视", "金运", "美视清", "智享屏",
        "没声音", "连不上", "卡顿", "闪退", "唱不了", "不能唱", "无法开机"
    ],
    "song": [
        "推荐", "歌曲", "歌", "想唱", "找歌", "歌手", "歌名", "周华健",
        "邓丽君", "Beyond", "合唱", "独唱", "练气", "音准"
    ],
    "sale": [
        "出货", "销量", "Q1", "Q2", "Q3", "Q4", "华东", "华南", "西南",
        "华北", "华中", "西北", "东北", "渠道", "售后率", "竞品", "份额",
        "销售", "卖了多少"
    ],
    "ticket": ["报修", "预约上门", "上门维修", "售后"]
}

TICKET_KEYWORDS = ["报修", "预约上门", "上门维修", "售后"]


# ========== 路由函数 ==========
def extract_tags(query: str) -> List[str]:
    query_lower = query.lower()
    matched_tags = []
    for tag, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in query_lower:
                matched_tags.append(tag)

    return matched_tags


def route_tag(query: str) -> Dict:
    query_lower = query.lower()

    # 第1优先级：ticket
    for kw in TICKET_KEYWORDS:
        if kw in query_lower:
            logger.info(f"命中工单关键词: {kw}")
            return {"type": "ticket", "tags": ["ticket"], "confidence": 0.95}

    # 第2优先级：device / song / sale
    matched_tags = extract_tags(query)
    unique_tags = list(set(matched_tags))
    tag_count = len(unique_tags)

    if tag_count == 1:
        return {"type": unique_tags[0], "tags": unique_tags, "confidence": 0.85}
    elif tag_count >= 2:
        return {"type": "composite", "tags": unique_tags, "confidence": 0.8}
    else:
        # 关键词无匹配，返回 None，让下一层处理
        return None


def llm_semantic_route(query: str) -> Dict:
    """大模型语义判断路由"""
    llm = init_chat_model(model="deepseek:deepseek-chat")
    prompt = f"""分析用户问题，判断属于以下哪一类，只输出一个词：

- device：设备问题（安装、连接、故障、黑屏、蓝牙、没声音）
- song：歌曲推荐（推荐歌、找歌手、练唱）
- sale：销售数据（销量、出货、渠道、区域业绩）
- composite：同时包含以上多种
- other：不属于以上任何一类

用户问题：{query}
输出："""
    response = llm.invoke(prompt)
    tag = response.content.strip().lower()

    if tag in ["device", "song", "sale", "composite"]:
        return {"type": tag, "tags": [tag], "confidence": 0.7, "source": "llm"}
    else:
        # 语义判断也不匹配，返回 other
        return {"type": "other", "tags": [], "confidence": 0.5, "source": "llm"}


def get_route_tag(query: str) -> Dict:
    # 第一层：关键词匹配
    result = route_tag(query)
    if result is not None:
        result["source"] = "keyword"
        return result

    # 第二层：大模型语义判断
    logger.info("关键词无匹配，调用大模型语义判断")
    result = llm_semantic_route(query)
    return result

# ========== 重排序检索（带超时和重试）==========
@retry(
    stop=stop_after_attempt(settings.rerank_max_retries + 1),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True
)


# ========== 创建混合检索器 ==========
def create_hybrid_retriever(kb, bm25_retriever, weights: list = [0.5, 0.5]):
    """创建 EnsembleRetriever 混合检索器"""
    if kb is None or bm25_retriever is None:
        return None

    # 将 FAISS 向量库包装成检索器
    vector_retriever = kb.as_retriever(search_kwargs={"k": 10})

    # 创建混合检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights
    )
    return ensemble_retriever

# 为三个库分别创建混合检索器
device_hybrid = create_hybrid_retriever(device_kb, device_bm25, weights=[0.7, 0.3])
song_hybrid = create_hybrid_retriever(song_kb, song_bm25, weights=[0.4, 0.6])
sales_hybrid = create_hybrid_retriever(sales_kb, sales_bm25, weights=[0.5, 0.5])


def extract_song_key(text: str) -> str:
    """提取歌曲唯一标识"""
    return text[:30]


def search_with_rerank(hybrid_retriever, query: str, fallback_answer: str, top_n: int = 5,
                       score_threshold: float = 0.6) -> str:
    """使用混合检索器 + 重排序"""
    if hybrid_retriever is None:
        return fallback_answer

    results = hybrid_retriever.invoke(query)
    if not results:
        return fallback_answer

    # ========== 去重：只保留前30个字符不同的文档 ==========
    seen = set()
    unique_results = []
    for doc in results:
        key = doc.page_content[:30]  # 取前30字符作为唯一标识
        if key not in seen:
            seen.add(key)
            unique_results.append(doc)
    results = unique_results
    # ========== 去重结束 ==========

    documents = [doc.page_content for doc in results[:30]]
    # 2. 准备文档
    documents = [doc.page_content for doc in results[:30]]

    try:
        # 3. 重排序
        rerank_response = rerank_client.post(
            "/reranks",
            body={
                "model": "qwen3-rerank",
                "query": query,
                "documents": documents,
                "top_n": len(documents),
                "instruct": "给定一个搜索查询，找出能够回答该问题的最相关文档片段。"
            },
            cast_to=object
        )

        if isinstance(rerank_response, dict) and rerank_response.get('results'):

            # 4. 按分数过滤
            filtered = []
            for item in rerank_response['results']:
                if item['relevance_score'] >= score_threshold:
                    idx = item['index']
                    filtered.append(documents[idx])


            if not filtered:
                best_score = rerank_response['results'][0]['relevance_score'] if rerank_response['results'] else 0
                return f"抱歉，没有找到与「{query}」足够相关的信息，请尝试其他关键词。"
            return "\n\n---\n\n".join(filtered[:top_n])
        else:
            return "\n\n---\n\n".join(documents[:top_n])

    except Exception as e:
        return "\n\n---\n\n".join(documents[:top_n])
# ========== 工具定义 ==========
@tool
def search_device_guide(query: str) -> str:
    """用户咨询家用K歌设备的安装、连接、功能使用、故障排查问题时，使用此工具。"""
    return search_with_rerank(device_hybrid, query, NO_RESULT_DEVICE, top_n=1)


@tool
def searsh_songs(query: str) -> str:
    """当用户希望推荐歌曲、根据场合/难度/风格选歌、获取练唱建议时，使用此工具。"""
    return search_with_rerank(song_hybrid, query, NO_RESULT_SONG, top_n=3)


@tool
def query_sales_data(query: str) -> str:
    """当用户查询出货量、销售数据、渠道商表现、区域业绩时，使用此工具。"""
    return search_with_rerank(sales_hybrid, query, NO_RESULT_SALE, top_n=1)


@tool
def create_service_ticket(query: str) -> str:
    """售后工单工具：当用户明确要求报修、预约上门、售后时使用"""
    query_lower = query.lower()

    # 安全校验
    SAFE_KEYWORDS = ["报修", "预约上门", "上门维修", "售后"]
    DEVICE_WORDS = ["维修手册", "排查步骤", "怎么办", "怎么解决", "故障排查"]

    has_safe_keyword = any(kw in query_lower for kw in SAFE_KEYWORDS)
    has_device_word = any(kw in query_lower for kw in DEVICE_WORDS)

    if not has_safe_keyword or has_device_word:
        return """【系统提示】❌ 无法生成工单

如需报修或售后，请明确说明：
- "我要报修"
- "预约上门维修"
- "售后问题需要处理"

当前问题未检测到明确的报修意图，请重新描述。"""

    # 识别设备
    device_type = "未知设备"
    if "机顶盒" in query_lower or "金运" in query_lower:
        device_type = "金运机顶盒"
    elif "音响" in query_lower or "美视清" in query_lower:
        device_type = "美视清智能音响"
    elif "电视" in query_lower or "智享屏" in query_lower:
        device_type = "智享屏"
    elif "麦克风" in query_lower:
        device_type = "无线麦克风"

    # 识别故障
    fault_desc = "设备故障"
    if "黑屏" in query_lower:
        fault_desc = "黑屏无显示"
    elif "蓝牙" in query_lower or "连不上" in query_lower:
        fault_desc = "蓝牙连接异常"
    elif "没声音" in query_lower:
        fault_desc = "无声音输出"
    elif "卡顿" in query_lower:
        fault_desc = "系统卡顿"

    # 生成工单
    ticket_id = f"LS{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"
    appointment_date = (datetime.now() + timedelta(days=2)).strftime("%Y年%m月%d日")

    return f"""【雷石售后工单系统】

工单号：{ticket_id}
设备：{device_type}
故障：{fault_desc}
状态：已受理

建议预约：{appointment_date}
客服电话：400-881-6666（提供工单号）

客服将在24小时内联系您确认上门时间。
"""


# ========== System Prompt ==========
system_prompt = """你是一个家用K歌智能助手，服务于雷石公司。

## 核心能力：自动拆分问题
当用户的问题包含多个子问题时，你必须：
1. 自动识别出所有子问题
2. 为每个子问题选择正确的工具
3. 依次调用工具获取答案
4. 最后整合成一个完整的回答

## 输出格式
复合问题时：
【问题拆分】
- 子问题1：xxx
- 子问题2：xxx

【执行步骤】
- 调用 [工具名] 查询子问题1
- 调用 [工具名] 查询子问题2

【回答】
**子问题1：**

**子问题2：**

简单问题时直接输出答案，不要输出【问题拆分】。
"""

# ========== Agent 初始化 ==========
os.makedirs("db", exist_ok=True)
conn = sqlite3.connect(settings.db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)

ls_agent = create_agent(
    model=model,
    tools=[search_device_guide, searsh_songs, query_sales_data, create_service_ticket],
    system_prompt=system_prompt
    ,checkpointer = checkpointer
)

config = {"configurable": {"thread_id": "user_001"}}


# ========== FastAPI 应用 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("雷石K歌智能助手启动中...")
    logger.info(f"向量库状态: device={device_kb is not None}, song={song_kb is not None}, sale={sales_kb is not None}")
    yield
    logger.info("应用关闭")


app = FastAPI(lifespan=lifespan)

# 限流配置
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== API 中间件 ==========
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """添加请求ID到日志"""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # 添加请求ID到日志上下文
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        return record

    logging.setLogRecordFactory(record_factory)

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(f"请求完成: {request.method} {request.url.path} | 耗时: {duration:.3f}s | 状态: {response.status_code}")
    return response


# ========== 并发控制信号量 ==========
semaphore = asyncio.Semaphore(settings.max_concurrent_requests)


def robust_search(query: str, route_type: str, history: List[str] = None) -> str:
    try:
        # 构建完整上下文
        full_context = ""
        if history:
            # 把最近三轮对话拼接起来
            full_context = " ".join(history[-3:])

        # 所有类型都走 Agent，带上完整上下文
        messages = []
        if full_context:
            messages.append({"role": "user", "content": f"历史对话：{full_context}"})
        messages.append({"role": "user", "content": query})

        result = ls_agent.invoke({"messages": messages}, config=config)
        return result['messages'][-1].content

    except Exception as e:
        logger.error(f"检索失败: {e}")
        return "系统繁忙，请稍后再试。"
# ========== 流式响应生成器 ==========
async def stream_generator(user_message: str, route: Dict):
    """带流式输出的响应生成器"""
    async with semaphore:  # 并发控制
        try:
            # 调用检索
            full = await asyncio.get_event_loop().run_in_executor(
                None, robust_search, user_message, route["type"]
            )

            # 流式输出
            for char in full:
                yield json.dumps({"token": char}) + "\n"
                await asyncio.sleep(0.02)
            yield json.dumps({"done": True}) + "\n"

        except asyncio.TimeoutError:
            yield json.dumps({"token": "请求超时，请稍后重试", "done": True}) + "\n"
        except Exception as e:
            logger.error(f"生成响应失败: {e}")
            yield json.dumps({"token": f"处理出错: {str(e)}", "done": True}) + "\n"


# ========== API 端点 ==========
@app.post("/chat/stream")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat_stream(request: Request, chat_req: ChatRequest):
    """聊天流式接口"""
    user_message = chat_req.message

    logger.info(f"用户问题: {user_message[:50]}...")
    route = get_route_tag(user_message)
    logger.info(f"路由结果: {route['type']}")

    return StreamingResponse(
        stream_generator(user_message, route),
        media_type="application/x-ndjson"
    )

# ========== 启动 ==========
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level=settings.log_level.lower()
    )