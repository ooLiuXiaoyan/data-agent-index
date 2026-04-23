#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库智能体 Mock API 服务
用于前端演示，无需 chromadb/torch 等重型依赖
"""

import os
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# 创建FastAPI应用
app = FastAPI(
    title="知识库智能体 API",
    description="基于ChromaDB的PDF向量化存储与检索服务 (Mock模式)",
    version="1.0.0-mock"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模拟数据
MOCK_KNOWLEDGE_BASE = [
    {
        "id": "chunk_001",
        "content": "测量标志保护改革的主要目标包括：建立动态巡查机制，实现测量标志的实时监控；推行\u201c电子身份证\u201d制度，为每个测量标志建立数字化档案；构建分级分类保护体系，根据标志的重要程度实施差异化保护措施。",
        "source": "浙江省测量标志保护改革技术方案.pdf",
        "page": 3,
        "distance": 0.92
    },
    {
        "id": "chunk_002",
        "content": "技术方案提出采用北斗高精度定位技术对测量标志进行精确定位，定位精度达到厘米级。同时利用无人机航测技术，定期对标志周边环境进行巡查，及时发现破坏和侵占行为。",
        "source": "浙江省测量标志保护改革技术方案.pdf",
        "page": 5,
        "distance": 0.88
    },
    {
        "id": "chunk_003",
        "content": "测量标志保护的责任主体包括：自然资源主管部门负责统筹协调，测绘资质单位负责日常维护，乡镇街道负责属地管理，村集体经济组织负责具体巡查。",
        "source": "浙江省测量标志保护改革技术方案.pdf",
        "page": 8,
        "distance": 0.85
    },
    {
        "id": "chunk_004",
        "content": "改革技术方案建议建设测量标志管理信息系统，集成标志档案管理、巡查监管、损毁报警、修复跟踪等功能模块，实现保护工作的信息化、智能化。",
        "source": "浙江省测量标志保护改革技术方案.pdf",
        "page": 12,
        "distance": 0.81
    },
    {
        "id": "chunk_005",
        "content": "浙江省现有各类测量标志约12万座，其中GNSS连续运行基准站85座、一等水准点320座、三角点5800座、GPS点86000座、水准点28000座。",
        "source": "浙江省测量标志保护改革技术方案.pdf",
        "page": 2,
        "distance": 0.78
    },
    {
        "id": "chunk_006",
        "content": "根据《中华人民共和国测绘法》和《浙江省测绘管理条例》，任何单位和个人不得损毁、擅自移动测量标志，不得占用永久性测量标志用地。",
        "source": "浙江省测量标志保护改革技术方案.pdf",
        "page": 15,
        "distance": 0.75
    },
    {
        "id": "chunk_007",
        "content": "技术方案明确：到2025年底，完成全省测量标志的普查建档工作；到2026年底，建立健全省市县三级联动保护机制；到2027年底，实现测量标志完好率95%以上的目标。",
        "source": "浙江省测量标志保护改革技术方案.pdf",
        "page": 4,
        "distance": 0.72
    },
    {
        "id": "chunk_008",
        "content": "测量标志的巡查频率要求：一等以上控制点每季度巡查一次，二等控制点每半年巡查一次，其他等级标志每年巡查一次。遇台风、洪水等自然灾害后应立即组织专项巡查。",
        "source": "浙江省测量标志保护改革技术方案.pdf",
        "page": 9,
        "distance": 0.70
    }
]

MOCK_FILES = [
    {"name": "浙江省测量标志保护改革技术方案.pdf", "size": "2.4 MB", "chunks": 48, "status": "已处理"},
    {"name": "测绘法律法规汇编.pdf", "size": "5.1 MB", "chunks": 96, "status": "已处理"},
    {"name": "测量标志维护技术规程.pdf", "size": "1.8 MB", "chunks": 36, "status": "已处理"}
]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/")
def root():
    return {"status": "ok", "mode": "mock", "message": "知识库智能体API服务运行中（演示模式）"}


@app.get("/api/files")
def get_files():
    return {"files": MOCK_FILES}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_size = random.randint(1024 * 1024, 10 * 1024 * 1024)
    chunks = random.randint(20, 100)
    return {
        "success": True,
        "filename": file.filename,
        "size": file_size,
        "chunks": chunks,
        "message": f"文件 {file.filename} 上传成功，已切分为 {chunks} 个文本块并向量化存储"
    }


@app.get("/api/stats")
def get_stats():
    return {
        "total_files": len(MOCK_FILES),
        "total_chunks": sum(f["chunks"] for f in MOCK_FILES),
        "total_vectors": sum(f["chunks"] for f in MOCK_FILES),
        "embedding_dim": 768,
        "model": "BAAI/bge-large-zh (Mock)",
        "collection": "zhejiang_survey_markers"
    }


@app.post("/api/search")
def search(req: SearchRequest):
    # 根据查询关键词简单排序模拟语义搜索
    query = req.query.lower()
    results = []
    for item in MOCK_KNOWLEDGE_BASE:
        score = random.uniform(0.65, 0.95)
        if any(kw in item["content"].lower() for kw in query.split()):
            score = min(score + 0.1, 0.98)
        results.append({**item, "distance": round(score, 4)})
    
    results.sort(key=lambda x: x["distance"], reverse=True)
    return {
        "query": req.query,
        "results": results[:req.top_k]
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    query = req.message.lower()
    
    # 简单的关键词匹配来生成回复
    if any(k in query for k in ["措施", "保护", "做法", "怎么"]):
        answer = """根据技术方案，测量标志保护的主要措施包括：

1. **动态巡查机制**：建立定期巡查制度，利用无人机和北斗定位技术实现精准监管
2. **电子身份证制度**：为每个测量标志建立数字化档案，包含位置、等级、状态等信息
3. **分级分类保护**：根据标志重要程度实施差异化保护，一等以上控制点每季度巡查一次
4. **信息化管理**：建设测量标志管理信息系统，集成档案管理、巡查监管、损毁报警等功能
5. **责任明确**：自然资源主管部门统筹协调，测绘资质单位日常维护，乡镇街道属地管理"""
    elif any(k in query for k in ["目标", "目的", "计划", "规划"]):
        answer = """技术方案明确了三个阶段的目标：

- **2025年底**：完成全省测量标志普查建档工作
- **2026年底**：建立健全省市县三级联动保护机制
- **2027年底**：实现测量标志完好率95%以上的目标

总体目标是构建\u201c天上看、地上查、网上管\u201d的立体化保护体系。"""
    elif any(k in query for k in ["现状", "数量", "多少", "统计"]):
        answer = """浙江省现有各类测量标志约**12万座**，具体分布如下：

- GNSS连续运行基准站：85座
- 一等水准点：320座
- 三角点：5,800座
- GPS点：86,000座
- 水准点：28,000座

目前面临的主要问题是部分标志损毁严重、保护责任不清、巡查手段落后等。"""
    elif any(k in query for k in ["技术", "方案", "方法", "北斗", "无人机"]):
        answer = """技术方案采用的关键技术包括：

1. **北斗高精度定位**：对测量标志进行厘米级精确定位
2. **无人机航测技术**：定期巡查标志周边环境，及时发现破坏行为
3. **物联网监测**：在重要标志上安装传感器，实时监测位移和环境变化
4. **移动巡查APP**：为巡查人员配备移动端应用，实现巡查轨迹记录和问题上报
5. **GIS可视化**：基于地理信息系统实现测量标志的空间展示和分析"""
    else:
        answer = f'您询问的是：\u201c{req.message}\u201d\n\n根据《浙江省测量标志保护改革技术方案》，这是一个涉及多部门协作的系统性工程。主要内容包括建立动态巡查机制、推行电子身份证制度、构建分级分类保护体系、建设信息化管理平台等。\n\n如需了解更具体的内容，您可以尝试询问：\n- 测量标志保护的主要措施有哪些？\n- 改革的目标和时间规划是什么？\n- 浙江省测量标志的现状如何？\n- 采用了哪些技术手段？'
    
    # 模拟检索到的相关文本块
    related = random.sample(MOCK_KNOWLEDGE_BASE, min(3, len(MOCK_KNOWLEDGE_BASE)))
    for r in related:
        r["distance"] = round(random.uniform(0.75, 0.93), 4)
    related.sort(key=lambda x: x["distance"], reverse=True)
    
    return {
        "answer": answer,
        "sources": related,
        "session_id": req.session_id or hashlib.md5(query.encode()).hexdigest()[:8]
    }


if __name__ == "__main__":
    print("🚀 启动知识库智能体 Mock API 服务...")
    print("📡 API地址: http://localhost:8000")
    print("📝 当前为演示模式，返回模拟数据")
    print("⚠️  如需完整功能，请在支持的环境中安装 chromadb + sentence-transformers")
    uvicorn.run(app, host="0.0.0.0", port=8000)
