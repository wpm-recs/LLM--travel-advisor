
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
    """生成集成模块 - 负责LLM集成问答生成"""

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
        将大模型的完整调用输入和输出记录到同路径的 log.txt 中
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
    def query_rewrite(self, query: str) -> str:
        """智能查询重写"""
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手，专门处理全球旅行相关搜索。请分析用户的查询，判断是否需要重写以提高检索效果。

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
根据用户的旅游问题，将其分类为以下两种类型之一：
1. 'detail' - 具体的路线、攻略
2. 'general' - 其他一般性问题

请只返回分类结果：detail 或 general

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

        if result in ['detail', 'general']:
            return result
        else:
            return 'general'

    def assess_context_relevance(self, query: str, retrieved_docs: List[Document]) -> bool:
        """Assess whether retrieved pages are relevant enough to be cited in the final answer."""
        if not retrieved_docs:
            return False

        context_preview = self._build_relevance_preview(retrieved_docs)
        prompt = ChatPromptTemplate.from_template("""
你是旅行问答系统的检索质量评估器。请判断给定的检索页面是否与用户问题足够相关，能够作为回答依据。

判断标准：
1. 如果页面直接回答问题，或明显属于同一地点/主题，返回 relevant。
2. 如果页面只有弱相关、主题偏移、地点错误，或不足以支撑回答，返回 irrelevant。
3. 不要因为有一点点关键词重合就判定为 relevant。

用户问题:
{question}

检索页面摘要:
{context_preview}

请只返回一个单词：relevant 或 irrelevant
""")

        full_prompt_text = prompt.format(question=query, context_preview=context_preview)
        chain = (
                {"question": RunnablePassthrough(), "context_preview": lambda _: context_preview}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()
        self._log_llm_interaction("检索相关性评估 (assess_context_relevance)", full_prompt_text, result)
        return result == "relevant"

    def generate_general_knowledge_answer(self, query: str, route_type: str = "general"):
        """Generate an answer using the model's own knowledge when retrieval is insufficient."""
        style_instruction = self._build_style_instruction(route_type)
        prompt = ChatPromptTemplate.from_template("""
你是一位专业旅行顾问。检索到的 Wikivoyage 页面不足以支持回答，因此你需要基于已有知识回答用户。

要求：
1. 回答开头必须明确写出：以下回答基于我的已有知识，而非检索到的 Wikivoyage 页面。
2. 不要假装引用了页面，也不要编造“根据页面所示”之类表述。
3. 如果某些信息你无法确定，要明确说明不确定。
4. {style_instruction}

用户问题: {question}

回答:
""")

        full_prompt_text = prompt.format(question=query, style_instruction=style_instruction)
        chain = (
                {"question": RunnablePassthrough(), "style_instruction": lambda _: style_instruction}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        full_response = ""
        for chunk in chain.stream(query):
            full_response += chunk
            yield chunk

        self._log_llm_interaction("已有知识回答 (generate_general_knowledge_answer)", full_prompt_text, full_response)

    def generate_basic_answer(self, query: str, context_docs: List[Document]):
        """生成基础回答 - 流式输出"""
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业导游。请根据以下指南信息回答用户的问题。

用户问题: {question}

相关指南信息:
{context}

请提供准确、友好的回答，完全基于相关指南给出的信息，不要使用已有知识。

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
        self._log_llm_interaction("基础回答(generate_basic_answer_stream)", full_prompt_text, full_response)

    def generate_step_by_step(self, query: str, context_docs: List[Document]):
        """生成详细攻略回答 - 流式输出"""
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的旅行规划师。请完全根据指南信息，为用户提供结构化的旅行攻略，不要使用已有知识。

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

    def _build_context(self, docs: List[Document], max_length: int = 500000) -> str:
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

    def _build_relevance_preview(self, docs: List[Document], max_docs: int = 5, max_chars: int = 1800) -> str:
        """Build a compact preview of retrieved pages for relevance assessment."""
        preview_parts = []
        current_length = 0

        for index, doc in enumerate(docs[:max_docs], 1):
            meta = doc.metadata
            title = meta.get('wiki_title') or meta.get('Title') or meta.get('relative_path') or meta.get('place_name') or 'Unknown'
            snippet = " ".join(doc.page_content.split())[:260]
            block = f"[{index}] 标题: {title}\n摘要: {snippet}\n"
            if current_length + len(block) > max_chars:
                break
            preview_parts.append(block)
            current_length += len(block)

        return "\n".join(preview_parts) if preview_parts else "无可用页面摘要"

    def _build_style_instruction(self, route_type: str) -> str:
        """Return style guidance based on the routed query type."""
        if route_type == "detail":
            return "请给出结构化、步骤化的旅行建议或行程安排。"
        return "请给出准确、直接、实用的旅行建议。"