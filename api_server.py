#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库智能体 API 服务
提供PDF上传、向量检索、智能问答接口
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# 检查依赖
try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    print("❌ 请先安装: pip3 install fastapi uvicorn")
    sys.exit(1)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("❌ 请先安装: pip3 install chromadb")
    sys.exit(1)

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("❌ 请先安装: pip3 install langchain langchain-community")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ 请先安装: pip3 install sentence-transformers")
    sys.exit(1)


# 创建FastAPI应用
app = FastAPI(
    title="知识库智能体 API",
    description="基于ChromaDB的PDF向量化存储与检索服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局配置
DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_db")
UPLOAD_DIR = DATA_DIR / "uploads"

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# 全局变量
embedding_model = None
chroma_client = None
collection = None


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str
    sources: List[dict]
    session_id: str


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    top_k: int = 5


def init_models():
    """初始化模型和数据库"""
    global embedding_model, chroma_client, collection
    
    print("🚀 正在初始化服务...")
    
    # 加载Embedding模型
    print("📥 加载 Embedding 模型 (BAAI/bge-large-zh)...")
    embedding_model = SentenceTransformer("BAAI/bge-large-zh")
    print(f"✅ 模型加载完成，维度: {embedding_model.get_sentence_embedding_dimension()}")
    
    # 初始化ChromaDB
    print("💾 初始化 ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # 获取或创建集合
    collection = chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={
            "hnsw:space": "cosine",
            "description": "知识库向量集合"
        }
    )
    
    doc_count = collection.count()
    print(f"✅ ChromaDB 准备就绪，当前文档数: {doc_count}")
    
    return embedding_model, collection


def process_pdf_to_chromadb(pdf_path: Path, collection_name: str = "knowledge_base") -> dict:
    """处理PDF并写入ChromaDB"""
    
    # 文本切分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    
    # 加载PDF
    print(f"📄 处理PDF: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    
    # 切分文本
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc.page_content))
    
    print(f"   生成 {len(chunks)} 个文本块")
    
    if not chunks:
        return {"success": False, "error": "PDF内容为空"}
    
    # 生成embeddings
    print("🔢 生成向量...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    
    # 准备数据
    file_hash = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
    ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": pdf_path.name,
            "source_type": "pdf",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "upload_time": datetime.now().isoformat()
        }
        for i in range(len(chunks))
    ]
    
    # 获取集合
    coll = chroma_client.get_or_create_collection(name=collection_name)
    
    # 添加到ChromaDB
    print("💾 写入数据库...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        coll.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=chunks[i:end],
            metadatas=metadatas[i:end]
        )
    
    return {
        "success": True,
        "chunks_count": len(chunks),
        "total_docs": coll.count(),
        "file_name": pdf_path.name
    }


@app.on_event("startup")
async def startup_event():
    """服务启动时初始化"""
    init_models()


@app.get("/")
async def root():
    """根路径"""
    return {
        "status": "running",
        "service": "知识库智能体 API",
        "docs": "/docs",
        "document_count": collection.count() if collection else 0
    }


@app.get("/api/files")
async def get_files():
    """获取已上传文件列表"""
    files = []
    
    # 从上传目录获取
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.suffix.lower() in ['.pdf', '.doc', '.docx', '.txt', '.py', '.js']:
                stat = f.stat()
                files.append({
                    "id": hashlib.md5(f.name.encode()).hexdigest()[:8],
                    "name": f.name,
                    "size": f"{stat.st_size / 1024:.1f} KB" if stat.st_size < 1024*1024 else f"{stat.st_size / 1024 / 1024:.1f} MB",
                    "type": f.suffix.lower().replace('.', ''),
                    "status": "已解析",
                    "upload_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                })
    
    # 从ChromaDB元数据获取
    try:
        if collection and collection.count() > 0:
            # 获取所有唯一文件名
            all_meta = collection.get()["metadatas"]
            db_files = set(m["source"] for m in all_meta if m)
            
            # 添加数据库中的文件（如果不在上传目录）
            for fname in db_files:
                if not any(f["name"] == fname for f in files):
                    files.append({
                        "id": hashlib.md5(fname.encode()).hexdigest()[:8],
                        "name": fname,
                        "size": "-",
                        "type": Path(fname).suffix.lower().replace('.', ''),
                        "status": "已入库",
                        "upload_time": "-"
                    })
    except Exception as e:
        print(f"获取数据库文件列表失败: {e}")
    
    return {"files": files}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件并处理"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    # 保存文件
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 如果是PDF，自动向量化
        if file.filename.lower().endswith('.pdf'):
            result = process_pdf_to_chromadb(file_path)
            return {
                "success": True,
                "message": "文件上传并处理成功",
                "file_name": file.filename,
                "processing_result": result
            }
        
        return {
            "success": True,
            "message": "文件上传成功",
            "file_name": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """智能问答"""
    
    if not collection or collection.count() == 0:
        return ChatResponse(
            answer="知识库为空，请先上传文档。",
            sources=[],
            session_id=request.session_id or "new"
        )
    
    try:
        # 生成查询向量
        query_embedding = embedding_model.encode([request.message]).tolist()
        
        # 检索相关文档
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        # 构建回答
        sources = []
        context_parts = []
        
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = (1 - dist) * 100
            if similarity > 50:  # 只使用相似度>50%的结果
                context_parts.append(doc)
                sources.append({
                    "file": meta.get('source', '未知'),
                    "chunk": meta.get('chunk_index', 0) + 1,
                    "total": meta.get('total_chunks', 0),
                    "similarity": f"{similarity:.1f}%",
                    "preview": doc[:100] + "..."
                })
        
        # 构建回答（简化版，不使用LLM）
        if sources:
            answer = f"根据知识库检索结果，为您找到 {len(sources)} 条相关内容：\n\n"
            for i, (doc, source) in enumerate(zip(context_parts[:3], sources[:3]), 1):
                answer += f"📄 **内容{i}**（相关度: {source['similarity']}）\n"
                answer += f"> {doc[:200]}...\n\n"
            answer += "如需更详细的分析，请提供更具体的问题。"
        else:
            answer = "未在知识库中找到相关内容。请尝试：\n1. 使用不同的关键词\n2. 上传更多相关文档\n3. 简化您的问题"
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id or hashlib.md5(request.message.encode()).hexdigest()[:8]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@app.post("/api/search")
async def search(request: SearchRequest):
    """语义搜索"""
    
    if not collection or collection.count() == 0:
        return {"results": [], "message": "知识库为空"}
    
    try:
        query_embedding = embedding_model.encode([request.query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=request.top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = (1 - dist) * 100
            formatted_results.append({
                "content": doc,
                "source": meta.get('source', '未知'),
                "chunk_index": meta.get('chunk_index', 0),
                "similarity": round(similarity, 2),
                "preview": doc[:150] + "..."
            })
        
        return {
            "query": request.query,
            "results": formatted_results,
            "total": len(formatted_results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    """获取知识库统计"""
    
    if not collection:
        return {"error": "服务未就绪"}
    
    count = collection.count()
    
    # 获取文件统计
    files = set()
    if count > 0:
        all_meta = collection.get()["metadatas"]
        for meta in all_meta:
            if meta and 'source' in meta:
                files.add(meta['source'])
    
    return {
        "total_documents": count,
        "unique_files": len(files),
        "files": list(files),
        "embedding_model": "BAAI/bge-large-zh",
        "vector_dimension": 768
    }


# 处理已存在的PDF文件
def process_existing_pdf():
    """处理已存在的PDF文件"""
    pdf_path = Path("~/Downloads/浙江省测量标志保护改革技术方案.pdf").expanduser()
    
    if pdf_path.exists():
        print(f"\n📄 发现PDF文件: {pdf_path.name}")
        
        # 复制到上传目录
        dest_path = UPLOAD_DIR / pdf_path.name
        import shutil
        shutil.copy2(pdf_path, dest_path)
        
        # 处理到ChromaDB
        result = process_pdf_to_chromadb(dest_path)
        
        if result["success"]:
            print(f"✅ 初始化完成，共 {result['total_docs']} 条向量")
        else:
            print(f"⚠️ 初始化失败: {result.get('error')}")
    else:
        print(f"⚠️ PDF文件不存在: {pdf_path}")


if __name__ == "__main__":
    import uvicorn
    
    # 初始化
    init_models()
    
    # 处理已存在的PDF
    process_existing_pdf()
    
    # 启动服务
    print("\n🌐 启动 API 服务...")
    print("   地址: http://localhost:8000")
    print("   文档: http://localhost:8000/docs")
    print("\n按 Ctrl+C 停止服务\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
