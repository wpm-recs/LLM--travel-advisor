"""
RAG系统主程序 - 全球旅行问答
"""

import os
import sys
import logging
from pathlib import Path
from typing import List
import pickle

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class TravelRAGSystem:

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")
        # 检查API密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化旅游RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化导游生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建旅游知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()
        chunks_file = self.config.chunks_path
        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            print("加载旅游指南文档...")
            print("📦 加载已保存的文本分块数据...")
            print("📦 加载已保存的文本分块数据...")
            with open(chunks_file, 'rb') as f:
                chunks = pickle.load(f)
        else:
            # 2. 加载文档
            print("加载旅游指南文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
            print("💾 保存文本分块数据...")
            with open(chunks_file, 'wb') as f:
                pickle.dump(chunks, f)

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)
            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息 (基于旅游数据元数据)
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats.get('total_documents', '未知')}")
        print(f"   文本块数: {stats.get('total_chunks', '未知')}")
        print("✅ 知识库构建完成！")

    def ask_question(self, question: str, stream: bool = True):
        """
        回答用户问题

        Args:
            question: 用户问题

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 旅客提问: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            rewritten_query = question
        else:
            print("🤖分析旅游需求...")
            rewritten_query = self.generation_module.query_rewrite(question)

        # 3. 检索相关子块（自动应用元数据过滤，如区域、类型）
        print("🔍检索相关旅游指南...")
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters,
                                                                             top_k=self.config.top_k)
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

        # 显示检索到的子块信息 (适配旅游元数据)
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                # 尝试获取具体地名，若无则取次级分类，最后取主区域
                chunk_path = chunk.metadata.get('relative_path')
                chunk_info.append(f"{chunk_path}")

            print(f"找到 {len(relevant_chunks)} 个相关攻略块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关攻略块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的旅游指南信息。您可以尝试换个城市或项目试试，比如'东京浅草附近有什么平价美食'。"

        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：直接返回推荐地点/餐厅列表
            print("📋 生成打卡点列表...")
            #relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            relevant_docs = relevant_chunks
            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # 详细查询：获取完整文档并生成详细攻略
            print("获取完整背景指南...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            print("✍️ 生成专业旅游攻略...")

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步攻略模式
                return self.generation_module.generate_step_by_step(question, relevant_docs)
            else:
                # 一般查询使用基础问答模式
                return self.generation_module.generate_basic_answer(question, relevant_docs)

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件 (适配旅游分类：餐饮、住宿、游览等)
        """
        filters = {}

        # 匹配 Section (大分类)
        section_mapping = {
            "Eat": ["吃", "美食", "餐厅", "餐饮", "小贩中心", "食阁"],
            "Sleep": ["住", "酒店", "住宿", "旅馆", "青年旅舍", "露营"],
            "See": ["看", "景点", "参观", "游览", "地标", "博物馆", "寺庙"],
            "Do": ["玩", "活动", "体验", "做"],
            "Buy": ["买", "购物", "商场", "伴手礼", "纪念品", "免税"],
            "Drink": ["喝", "酒吧", "夜生活", "夜店", "咖啡"],
            "Get in": ["交通", "入境", "机场", "过关", "怎么去"],
            "Stay safe": ["安全", "治安", "危险", "罚款", "禁止"],
        }


        # 检测分类
        for section, keywords in section_mapping.items():
            if any(keyword in query for keyword in keywords):
                filters['Section'] = section
                break

        return filters


    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 65)
        print(" 旅游指南助手")
        print("=" * 65)
        print("   (例如：'罗马有哪些适合步行的景点？', '巴黎地铁通票怎么买？')")

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        print("\n开始咨询 (输入 '退出' 或 'exit' 结束):")

        while True:
            try:
                user_input = input("\n🎒 您的提问: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break


                print("\n🧑‍💼 导游回答:")
                for chunk in self.ask_question(user_input, stream=True):
                    print(chunk, end="", flush=True)
                print("\n")

                # [Image tag generation rule applied here conceptually by the LLM response module]

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 处理问题时出错: {e}")



def main():
    """主函数"""
    try:
        # 创建全球旅行RAG系统
        rag_system = TravelRAGSystem()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()