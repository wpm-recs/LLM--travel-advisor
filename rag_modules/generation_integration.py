
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
You are an intelligent query analysis assistant for global travel search.
Analyze the user query and decide whether rewriting is needed to improve retrieval quality.

Original query: {query}

Rules:
1. If the query is already specific and clear, return it unchanged.
2. If the query is vague, ambiguous, or overly colloquial, rewrite it into a clearer travel-search query.

Output only the final query text (unchanged if no rewrite is needed):""",
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
Classify the user's travel question into exactly one of the following types:
1. 'detail' - specific itinerary, route, or actionable travel plan
2. 'general' - all other general travel questions

Return only one label: detail or general

User question: {query}

Classification:""")

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
You are a retrieval-quality evaluator for a travel Q&A system.
Decide whether the retrieved pages are relevant enough to support an answer.

Criteria:
1. Return relevant if pages directly answer the question or clearly match the same place/topic.
2. Return irrelevant if pages are only weakly related, off-topic, mismatched in location, or insufficient for grounding.
3. Do not mark relevant based on minor keyword overlap alone.

User question:
{question}

Retrieved page summaries:
{context_preview}

Return exactly one word: relevant or irrelevant
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
You are a professional travel advisor.
The retrieved Wikivoyage pages are not sufficient to ground the answer, so respond using your own general knowledge.

Requirements:
1. At the beginning, explicitly state that this answer is based on your own knowledge rather than retrieved Wikivoyage pages, phrased in the same language as the user question.
2. Do not pretend to cite retrieved pages or fabricate source-based claims.
3. If any detail is uncertain, clearly say so.
4. {style_instruction}
5. Respond in the same language as the user question.

User question: {question}

Answer:
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
You are a professional tour guide.
Answer the user's question based on the travel-guide information below.

Requirements:
1. Respond in the same language as the user question.
2. Keep the answer accurate and friendly.
3. Ground the answer strictly in the provided guide information, without using outside knowledge.

User question: {question}

Relevant guide information:
{context}

Answer:""")

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
You are a professional travel planner.
Using only the guide information below, provide a structured and actionable travel plan.

Requirements:
1. Respond in the same language as the user question.
2. Keep the plan clearly structured and directly actionable.
3. Do not use outside knowledge beyond the provided guide information.

User question: {question}

Relevant guide information:
{context}

Answer:""")

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
            return "Provide a structured, step-by-step travel plan with practical execution details."
        return "Provide accurate, direct, and practical travel advice."