# 知识库智能体系统 - 使用说明

## 📁 文件结构

```
knowledge-base-agent/
├── index.html              # 技术路线展示页面
├── demo.html               # 案例演示页面（使用真实数据）
├── api_server.py           # 后端API服务
├── pdf_to_chromadb.py      # PDF向量化工具（完整版）
├── simple_pdf_to_chromadb.py  # PDF向量化工具（精简版）
├── README.md               # 基础说明
└── SETUP.md               # 本文件
```

## 🚀 快速启动

### 步骤1: 安装依赖

等待以下依赖安装完成（可能需要几分钟）：

```bash
pip3 install chromadb langchain langchain-community sentence-transformers pypdf fastapi uvicorn
```

当前状态检查：
```bash
pip3 show chromadb
```

### 步骤2: 启动API服务

```bash
cd /Users/mushroom/Desktop/Coder/data-agent-index/knowledge-base-agent
python3 api_server.py
```

服务启动后会：
1. 自动加载 Embedding 模型 (BAAI/bge-large-zh)
2. 初始化 ChromaDB 数据库
3. 自动处理 PDF 文件：~/Downloads/浙江省测量标志保护改革技术方案.pdf
4. 启动 API 服务：http://localhost:8000

### 步骤3: 打开演示页面

在浏览器中打开：
- 技术路线：http://localhost:8001/index.html （或直接打开文件）
- 案例演示：http://localhost:8001/demo.html （需要先启动API服务）

或者使用Python启动本地服务器：
```bash
python3 -m http.server 8001
```

## 📡 API接口说明

### 1. 获取文件列表
```http
GET /api/files
```

### 2. 上传文件
```http
POST /api/upload
Content-Type: multipart/form-data

file: <文件>
```

### 3. 智能问答
```http
POST /api/chat
Content-Type: application/json

{
  "message": "测量标志保护的主要措施有哪些？",
  "session_id": "optional"
}
```

### 4. 语义搜索
```http
POST /api/search
Content-Type: application/json

{
  "query": "技术方案",
  "top_k": 5
}
```

### 5. 获取统计信息
```http
GET /api/stats
```

## 🔧 系统架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   前端页面       │────▶│   API服务       │────▶│   ChromaDB      │
│                 │     │                 │     │                 │
│  - index.html   │     │  - FastAPI      │     │  - 向量存储      │
│  - demo.html    │     │  - 语义检索      │     │  - HNSW索引      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  Embedding模型   │
                        │                 │
                        │  BAAI/bge-large │
                        │  -zh (768维)    │
                        └─────────────────┘
```

## 📊 数据处理流程

```
PDF文件
   │
   ▼
PyPDFLoader 提取文本
   │
   ▼
RecursiveCharacterTextSplitter (500字符/块, 50字符重叠)
   │
   ▼
SentenceTransformer (BAAI/bge-large-zh) 生成向量
   │
   ▼
ChromaDB 存储 + HNSW索引
   │
   ▼
语义检索 (Cosine相似度)
```

## 💡 使用示例

### 测试查询

启动服务后，可以在 demo.html 页面中输入：

1. **文档相关问题**：
   - "测量标志保护的主要措施有哪些？"
   - "技术方案的改革目标是什么？"
   - "浙江省测量标志的现状如何？"

2. **语义搜索**：
   - 系统会自动检索知识库中最相关的文本块
   - 显示相似度百分比
   - 标注来源文件

### 上传新文件

1. 在 demo.html 页面点击上传区域
2. 选择 PDF 文件
3. 系统自动完成：
   - 文件保存到 data/uploads/
   - 文本提取和切分
   - 向量化处理
   - 写入 ChromaDB

## 🔍 故障排查

### API未连接
检查API服务是否运行：
```bash
curl http://localhost:8000/
```

### 依赖缺失
```bash
pip3 install chromadb langchain langchain-community sentence-transformers pypdf fastapi uvicorn --upgrade
```

### 模型下载失败
首次运行需要下载Embedding模型（约1GB）：
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-zh")
```

## 📈 性能指标

- **向量维度**: 768 (BGE-large-zh)
- **切分策略**: 500字符/块，50字符重叠
- **索引类型**: HNSW (Hierarchical Navigable Small World)
- **相似度算法**: Cosine
- **检索延迟**: < 100ms (万级向量)

## 📝 待办事项

- [ ] 依赖安装完成
- [ ] 启动API服务测试
- [ ] 验证PDF向量化
- [ ] 前端页面联调
- [ ] 添加更多文件类型支持
