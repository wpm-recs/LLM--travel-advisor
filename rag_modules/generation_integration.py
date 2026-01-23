"""
生成集成模块 - 适配新加坡旅游指南检索系统
"""

import os
import logging
from datetime import datetime
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和新加坡旅游问答生成"""

    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            moonshot_api_key=api_key
        )

        logger.info("LLM初始化完成")

    # ================= LLM 调用日志记录方法 =================
    def _log_llm_interaction(self, method_name: str, full_prompt: str, output_text: str):
        """
        将大模型的完整调用输入（模板填充后）和输出记录到同路径的 log.txt 中
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(current_dir, "log.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_content = (
            f"\n{'='*70}\n"
            f"时间: {timestamp} | 调用方法: {method_name}\n"
            f"【完整输入 (Filled Prompt)】:\n{full_prompt}\n"
            f"{'-'*30}\n"
            f"【输出 / 解答】:\n{output_text}\n"
            f"{'='*70}\n"
        )

        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(log_content)
        except Exception as e:
            logger.error(f"写入日志文件失败: {e}")
    # ============================================================

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """生成基础回答"""
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位精通新加坡的专业导游。请根据以下《新加坡旅游指南》信息回答用户的问题。

用户问题: {question}

相关指南信息:
{context}

请提供准确、友好的回答。如果指南信息中没有相关答案，请结合上下文诚实说明，不要编造。
回答需符合中文语言习惯，提及地点时如有英文原名可括号保留（如 Orchard Road）。

回答:""")

        # 显式生成完整的 Prompt 文本用于日志
        full_prompt_text = prompt.format(question=query, context=context)

        chain = (
                {"question": RunnablePassthrough(), "context": lambda _: context}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        response = chain.invoke(query)

        # 记录日志：传入完整的 full_prompt_text
        self._log_llm_interaction("基础回答 (generate_basic_answer)", full_prompt_text, response)

        return response

    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """生成结构化/分步骤的详细旅游攻略回答"""
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的新加坡旅行规划师。请根据指南信息，为用户提供结构化、详细的旅行攻略或分步指南。

用户问题: {question}

相关指南信息:
{context}

请灵活组织回答，建议包含以下部分（根据实际查询内容调整）：

## 📍 地点/主题概述
## 🚶‍♂️ 交通指南 (Get around)
## 🌟 推荐体验/打卡点 (See & Do)
## 💡 实用贴士 (Tips/Safety)

回答:""")

        # 显式生成完整的 Prompt 文本用于日志
        full_prompt_text = prompt.format(question=query, context=context)

        chain = (
                {"question": RunnablePassthrough(), "context": lambda _: context}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        response = chain.invoke(query)

        # 记录日志
        self._log_llm_interaction("详细攻略 (generate_step_by_step_answer)", full_prompt_text, response)

        return response

    def query_rewrite(self, query: str) -> str:
        """智能查询重写"""
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手，专门处理有关新加坡旅游的搜索。请分析用户的查询，判断是否需要重写以提高检索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）
2. **模糊不清或过于口语化的查询**（需要重写）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        # 显式生成完整的 Prompt 文本用于日志
        full_prompt_text = prompt.format(query=query)

        chain = (
                {"query": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录日志
        self._log_llm_interaction("查询重写 (query_rewrite)", full_prompt_text, response)

        return response

    def query_router(self, query: str) -> str:
        """查询路由"""
        prompt = ChatPromptTemplate.from_template("""
根据用户的旅游问题，将其分类为以下三种类型之一：
1. 'list' - 获取推荐列表
2. 'detail' - 具体的路线、攻略
3. 'general' - 其他一般性问题

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        # 显式生成完整的 Prompt 文本用于日志
        full_prompt_text = prompt.format(query=query)

        chain = (
                {"query": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        # 记录日志
        self._log_llm_interaction("查询路由 (query_router)", full_prompt_text, result)

        if result in ['list', 'detail', 'general']:
            return result
        else:
            return 'general'

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """生成列表式回答 (纯逻辑处理，未调用LLM)"""
        # ... (此方法未调用 LLM，保持不变)
        if not context_docs:
            return "抱歉，指南中暂无相关的推荐信息。"

        items = []
        for doc in context_docs:
            item_name = doc.metadata.get('Item_Name') or doc.metadata.get('Sub_Section') or doc.metadata.get('Title', '未知地点')
            if item_name not in items:
                items.append(item_name)

        if len(items) == 1:
            return f"为您推荐：{items[0]}"
        elif len(items) <= 5:
            return f"为您精选以下推荐：\n" + "\n".join([f"{i + 1}. {name}" for i, name in enumerate(items)])
        else:
            return f"为您精选以下热门推荐：\n" + "\n".join([f"{i + 1}. {name}" for i, name in enumerate(items[:5])]) + f"\n\n此外，还有其他 {len(items) - 5} 个选择可供探索。"

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """生成基础回答 - 流式输出"""
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位精通新加坡的专业导游。请根据以下《新加坡旅游指南》信息回答用户的问题。

用户问题: {question}

相关指南信息:
{context}

请提供准确、友好的回答。

回答:""")

        # 显式生成完整的 Prompt 文本用于日志
        full_prompt_text = prompt.format(question=query, context=context)

        chain = (
                {"question": RunnablePassthrough(), "context": lambda _: context}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        full_response = ""
        for chunk in chain.stream(query):
            full_response += chunk
            yield chunk

        # 流式传输结束后，记录完整 Prompt 和完整回答
        self._log_llm_interaction("基础回答[流式] (generate_basic_answer_stream)", full_prompt_text, full_response)

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """生成详细攻略回答 - 流式输出"""
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的新加坡旅行规划师。请根据指南信息，为用户提供结构化的旅行攻略。

用户问题: {question}

相关指南信息:
{context}

回答:""")

        # 显式生成完整的 Prompt 文本用于日志
        full_prompt_text = prompt.format(question=query, context=context)

        chain = (
                {"question": RunnablePassthrough(), "context": lambda _: context}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        full_response = ""
        for chunk in chain.stream(query):
            full_response += chunk
            yield chunk

        # 流式传输结束后，记录完整 Prompt 和完整回答
        self._log_llm_interaction("详细攻略[流式] (generate_step_by_step_answer_stream)", full_prompt_text, full_response)

    def _build_context(self, docs: List[Document], max_length: int = 2500) -> str:
        """构建上下文字符串"""
        if not docs:
            return "暂无相关旅游指南信息。"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            category = meta.get('category', '一般信息')
            title = meta.get('Title', '')
            section = meta.get('Section', '')
            sub_section = meta.get('Sub_Section', '')
            item_name = meta.get('Item_Name', '')

            hierarchy_parts = [p for p in [title, section, sub_section, item_name] if p]
            hierarchy = " > ".join(hierarchy_parts)

            metadata_info = f"【指南 {i}】 {hierarchy} | 分类: {category}"

            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n" + "=" * 50 + "\n".join(context_parts)