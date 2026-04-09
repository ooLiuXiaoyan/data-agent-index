#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF文件向量化存储到ChromaDB
支持自动文本切分、Embedding生成和索引建立
"""

import os
import sys
from pathlib import Path
from typing import List

# 检查必要依赖
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("❌ 请先安装 chromadb: pip install chromadb")
    sys.exit(1)

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("❌ 请先安装 langchain: pip install langchain langchain-community")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ 请先安装 sentence-transformers: pip install sentence-transformers")
    sys.exit(1)


class PDFToChromaDB:
    """PDF文件向量化处理器"""
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "BAAI/bge-large-zh",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化处理器
        
        Args:
            collection_name: ChromaDB集合名称
            persist_directory: 数据持久化目录
            embedding_model: 嵌入模型名称
            chunk_size: 文本切分块大小
            chunk_overlap: 文本块重叠大小
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化 Embedding 模型
        print(f"📥 正在加载 Embedding 模型: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✅ 模型加载完成")
        
        # 初始化 ChromaDB 客户端
        self._init_chromadb()
        
        # 初始化文本切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", " ", ""]
        )
    
    def _init_chromadb(self):
        """初始化ChromaDB客户端和集合"""
        # 确保持久化目录存在
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 创建客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合（自动建立索引）
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",  # 使用余弦相似度
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 128,
                "hnsw:M": 16,
                "description": "知识库向量集合"
            }
        )
        
        print(f"📦 ChromaDB 集合 '{self.collection_name}' 准备就绪")
        print(f"   存储路径: {os.path.abspath(self.persist_directory)}")
        print(f"   当前文档数: {self.collection.count()}")
    
    def load_pdf(self, pdf_path: str) -> List[str]:
        """
        加载PDF文件并切分文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            切分后的文本块列表
        """
        pdf_path = Path(pdf_path).expanduser()
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        print(f"\n📄 正在加载PDF: {pdf_path.name}")
        
        # 使用 LangChain 加载PDF
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        print(f"   共 {len(documents)} 页")
        
        # 合并所有页面文本
        full_text = "\n\n".join([doc.page_content for doc in documents])
        print(f"   总字符数: {len(full_text)}")
        
        # 文本切分
        print(f"\n✂️ 正在切分文本 (块大小: {self.chunk_size}, 重叠: {self.chunk_overlap})")
        chunks = self.text_splitter.split_text(full_text)
        print(f"   生成 {len(chunks)} 个文本块")
        
        # 显示前3个文本块示例
        for i, chunk in enumerate(chunks[:3], 1):
            preview = chunk[:80].replace("\n", " ")
            print(f"   块{i}: {preview}...")
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        print(f"\n🔢 正在生成 Embedding 向量...")
        print(f"   模型: {self.embedding_model.get_sentence_embedding_dimension()}维")
        
        # 批量生成embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        print(f"   共生成 {len(embeddings)} 个向量")
        return embeddings.tolist()
    
    def add_to_chromadb(
        self,
        chunks: List[str],
        pdf_path: str,
        source_type: str = "pdf"
    ):
        """
        将文本块添加到ChromaDB
        
        Args:
            chunks: 文本块列表
            pdf_path: PDF文件路径（用于元数据）
            source_type: 数据源类型
        """
        if not chunks:
            print("⚠️ 没有文本块需要添加")
            return
        
        pdf_name = Path(pdf_path).name
        
        # 生成embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # 准备数据
        ids = [f"{pdf_name}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": pdf_name,
                "source_type": source_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
            for i in range(len(chunks))
        ]
        
        print(f"\n💾 正在写入 ChromaDB...")
        
        # 分批添加（避免单次请求过大）
        batch_size = 100
        total = len(chunks)
        
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=chunks[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            print(f"   进度: {batch_end}/{total}")
        
        print(f"✅ 成功写入 {total} 条记录")
    
    def process_pdf(self, pdf_path: str) -> dict:
        """
        处理PDF完整流程
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            处理结果统计
        """
        print("=" * 60)
        print("🚀 开始 PDF 向量化处理")
        print("=" * 60)
        
        try:
            # 1. 加载并切分PDF
            chunks = self.load_pdf(pdf_path)
            
            if not chunks:
                return {"success": False, "error": "PDF内容为空"}
            
            # 2. 添加到ChromaDB
            self.add_to_chromadb(chunks, pdf_path)
            
            # 3. 统计信息
            total_docs = self.collection.count()
            
            print("\n" + "=" * 60)
            print("📊 处理完成统计")
            print("=" * 60)
            print(f"   PDF文件: {Path(pdf_path).name}")
            print(f"   文本块数: {len(chunks)}")
            print(f"   向量维度: {self.embedding_model.get_sentence_embedding_dimension()}")
            print(f"   集合总数: {total_docs}")
            print(f"   存储路径: {os.path.abspath(self.persist_directory)}")
            print("=" * 60)
            
            return {
                "success": True,
                "pdf_name": Path(pdf_path).name,
                "chunks_count": len(chunks),
                "total_docs": total_docs,
                "embedding_dim": self.embedding_model.get_sentence_embedding_dimension()
            }
            
        except Exception as e:
            print(f"\n❌ 处理失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def search(self, query: str, n_results: int = 5) -> dict:
        """
        语义搜索
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            搜索结果
        """
        print(f"\n🔍 搜索: {query}")
        
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"   找到 {len(results['documents'][0])} 条相关结果\n")
        
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            similarity = (1 - dist) * 100
            preview = doc[:150].replace("\n", " ")
            print(f"{i}. [{similarity:.1f}%] {preview}...")
            print(f"   来源: {meta['source']} (块{meta['chunk_index']+1}/{meta['total_chunks']})\n")
        
        return results
    
    def delete_collection(self):
        """删除集合（谨慎使用）"""
        confirm = input(f"⚠️ 确认删除集合 '{self.collection_name}'? (输入 'yes' 确认): ")
        if confirm == "yes":
            self.client.delete_collection(self.collection_name)
            print(f"✅ 集合 '{self.collection_name}' 已删除")
        else:
            print("已取消删除操作")


def main():
    """主函数"""
    # PDF文件路径（下载目录）
    pdf_path = "~/Downloads/浙江省测量标志保护改革技术方案.pdf"
    
    # 展开用户目录
    pdf_path = os.path.expanduser(pdf_path)
    
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        print("请确认文件已下载到下载目录")
        return
    
    # 初始化处理器
    processor = PDFToChromaDB(
        collection_name="zhejiang_survey_markers",
        persist_directory="./chroma_db",
        embedding_model="BAAI/bge-large-zh",  # 中文优化模型
        chunk_size=500,    # 每块500字符
        chunk_overlap=50   # 重叠50字符保证语义连贯
    )
    
    # 处理PDF
    result = processor.process_pdf(pdf_path)
    
    if result["success"]:
        # 测试搜索
        print("\n💡 输入查询语句测试语义搜索 (输入 'quit' 退出):")
        while True:
            query = input("\n查询: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                processor.search(query)
    
    print("\n✨ 处理完成！")


if __name__ == "__main__":
    main()
