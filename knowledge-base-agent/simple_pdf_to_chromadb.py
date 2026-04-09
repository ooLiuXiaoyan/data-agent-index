#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF文件向量化存储到ChromaDB - 精简版
"""

import os
import sys
from pathlib import Path

# PDF文件路径
PDF_PATH = os.path.expanduser("~/Downloads/浙江省测量标志保护改革技术方案.pdf")

def check_dependencies():
    """检查依赖"""
    missing = []
    
    try:
        import chromadb
        print(f"✅ chromadb: {chromadb.__version__}")
    except ImportError:
        missing.append("chromadb")
    
    try:
        import langchain
        print(f"✅ langchain: 已安装")
    except ImportError:
        missing.append("langchain")
    
    try:
        import langchain_community
        print(f"✅ langchain-community: 已安装")
    except ImportError:
        missing.append("langchain-community")
    
    try:
        import sentence_transformers
        print(f"✅ sentence-transformers: 已安装")
    except ImportError:
        missing.append("sentence-transformers")
    
    try:
        import pypdf
        print(f"✅ pypdf: 已安装")
    except ImportError:
        missing.append("pypdf")
    
    if missing:
        print(f"\n❌ 缺少依赖: {', '.join(missing)}")
        print("\n请运行以下命令安装:")
        print("pip3 install " + " ".join(missing))
        return False
    
    return True

def process_pdf():
    """处理PDF主流程"""
    
    print("=" * 60)
    print("🚀 PDF 向量化处理工具")
    print("=" * 60)
    
    # 检查文件
    if not os.path.exists(PDF_PATH):
        print(f"❌ 文件不存在: {PDF_PATH}")
        return
    
    file_size = os.path.getsize(PDF_PATH) / 1024 / 1024
    print(f"📄 PDF文件: {Path(PDF_PATH).name}")
    print(f"   大小: {file_size:.2f} MB")
    
    # 导入依赖
    from chromadb.config import Settings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    import chromadb
    
    # 配置参数
    COLLECTION_NAME = "zhejiang_survey_markers"
    PERSIST_DIR = "./chroma_db"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # 1. 加载PDF
    print("\n📖 正在读取PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"   共 {len(documents)} 页")
    
    # 2. 切分文本
    print("\n✂️ 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc.page_content))
    
    print(f"   生成 {len(chunks)} 个文本块")
    
    # 显示前3个文本块预览
    for i, chunk in enumerate(chunks[:3], 1):
        preview = chunk[:60].replace("\n", " ")
        print(f"   块{i}: {preview}...")
    
    # 3. 加载Embedding模型
    print("\n🔢 正在加载 Embedding 模型 (BAAI/bge-large-zh)...")
    print("   首次加载需要下载模型，请耐心等待...")
    model = SentenceTransformer("BAAI/bge-large-zh")
    print(f"   ✅ 模型加载完成，维度: {model.get_sentence_embedding_dimension()}")
    
    # 4. 生成向量
    print("\n📊 正在生成向量...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"   共生成 {len(embeddings)} 个向量")
    
    # 5. 初始化ChromaDB
    print("\n💾 正在初始化 ChromaDB...")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # 获取或创建集合（自动建立索引）
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "description": "浙江省测量标志保护改革技术方案"
        }
    )
    
    # 6. 写入数据
    print(f"\n💿 正在写入数据到集合 '{COLLECTION_NAME}'...")
    
    ids = [f"doc_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": Path(PDF_PATH).name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        for i in range(len(chunks))
    ]
    
    # 分批添加
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=chunks[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"   进度: {end}/{len(chunks)}")
    
    # 7. 完成统计
    print("\n" + "=" * 60)
    print("✅ 处理完成!")
    print("=" * 60)
    print(f"📁 集合名称: {COLLECTION_NAME}")
    print(f"📊 文档总数: {collection.count()}")
    print(f"🔢 向量维度: {model.get_sentence_embedding_dimension()}")
    print(f"💾 存储路径: {os.path.abspath(PERSIST_DIR)}")
    print("=" * 60)
    
    # 8. 测试查询
    print("\n💡 测试语义搜索:")
    test_queries = [
        "测量标志保护",
        "技术方案",
        "改革措施"
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: '{query}'")
        query_vec = model.encode([query])
        results = collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=2
        )
        for doc, dist in zip(results['documents'][0], results['distances'][0]):
            similarity = (1 - dist) * 100
            preview = doc[:80].replace("\n", " ")
            print(f"   [{similarity:.1f}%] {preview}...")
    
    return collection

if __name__ == "__main__":
    print("正在检查依赖...\n")
    if check_dependencies():
        print("\n所有依赖已安装，开始处理...\n")
        try:
            process_pdf()
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        sys.exit(1)
