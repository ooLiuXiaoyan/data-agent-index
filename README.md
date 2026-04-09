# PDF向量化存储到ChromaDB

## 脚本说明

已创建两个版本的脚本：

1. **`simple_pdf_to_chromadb.py`** - 精简版（推荐）
2. **`pdf_to_chromadb.py`** - 完整版（包含更多功能）

## 快速开始

### 1. 安装依赖

```bash
pip3 install chromadb langchain langchain-community sentence-transformers pypdf
```

### 2. 运行脚本

```bash
cd /Users/mushroom/Desktop/Coder/data-agent-index/knowledge-base-agent
python3 simple_pdf_to_chromadb.py
```

## 功能特性

- ✅ 自动读取PDF文件（支持多页）
- ✅ 智能文本切分（500字符/块，50字符重叠）
- ✅ 中文优化Embedding模型（BAAI/bge-large-zh）
- ✅ 自动向量化（768维）
- ✅ ChromaDB存储（自动建立HNSW索引）
- ✅ 持久化存储到 `./chroma_db` 目录
- ✅ 内置语义搜索测试

## 处理流程

```
PDF文件 → 文本提取 → 智能切分 → Embedding生成 → ChromaDB存储 → 自动索引
```

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| collection_name | 集合名称 | zhejiang_survey_markers |
| chunk_size | 切分块大小 | 500字符 |
| chunk_overlap | 重叠大小 | 50字符 |
| embedding_model | 嵌入模型 | BAAI/bge-large-zh |
| persist_directory | 存储目录 | ./chroma_db |

## 索引说明

ChromaDB自动使用 **HNSW（Hierarchical Navigable Small World）** 索引算法：

- **索引类型**: Approximate Nearest Neighbor (ANN)
- **相似度度量**: Cosine（余弦相似度）
- **搜索复杂度**: O(log n)
- **特点**: 高性能、内存友好、支持增量更新

## 使用示例

### 基础使用
```python
from simple_pdf_to_chromadb import process_pdf

# 处理PDF
collection = process_pdf()

# 语义搜索
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5
)
```

### 手动搜索
运行脚本后会自动进入交互式搜索模式，输入查询即可测试语义搜索效果。

## 数据存储位置

- **向量数据**: `./chroma_db/` 目录
- **可备份、可复制**
- **支持多集合管理**

## 待安装依赖安装完成后即可运行

当前状态: 依赖安装中...（正在编译numpy）
