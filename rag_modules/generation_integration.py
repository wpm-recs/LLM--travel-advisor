"""
生成集成模块 - 适配新加坡旅游指南检索系统
"""

import os
import logging
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
        """
        初始化生成集成模块

        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
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

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位精通新加坡的专业导游。请根据以下《新加坡旅游指南》信息回答用户的问题。

用户问题: {question}

相关指南信息:
{context}

请提供准确、友好的回答。如果指南信息中没有相关答案，请结合上下文诚实说明，不要编造。
回答需符合中文语言习惯，提及地点时如有英文原名可括号保留（如 Orchard Road）。

回答:""")

        # 使用LCEL构建链
        chain = (
                {"question": RunnablePassthrough(), "context": lambda _: context}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成结构化/分步骤的详细旅游攻略回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            结构化的详细旅游指南
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的新加坡旅行规划师。请根据指南信息，为用户提供结构化、详细的旅行攻略或分步指南。

用户问题: {question}

相关指南信息:
{context}

请灵活组织回答，建议包含以下部分（根据实际查询内容调整）：

## 📍 地点/主题概述
[简要介绍该景点、街区或体验的核心魅力和特色]

## 🚶‍♂️ 交通指南 (Get around)
[详细说明如何到达该地，包括建议的MRT地铁站、巴士路线或步行指引]

## 🌟 推荐体验/打卡点 (See & Do)
[分点列出值得游览的具体项目、商场或餐厅，并附上简要评价或消费提示]

## 💡 实用贴士 (Tips/Safety)
[列出相关的文化礼仪(Respect)、安全事项(Stay safe)或最佳游览时间等]

注意：
- 结构要清晰，重点突出，方便游客手机端快速阅读。
- 重点突出实用性，例如价格、开放时间、禁忌等信息（如果原文有提及）。
- 若信息不足，省略相应板块，不要强行填充。

回答:""")

        chain = (
                {"question": RunnablePassthrough(), "context": lambda _: context}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询以提高检索效果

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手，专门处理有关新加坡旅游的搜索。请分析用户的查询，判断是否需要重写以提高检索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体地点或项目：如"乌节路购物攻略"、"滨海湾花园门票"、"小印度怎么去"
   - 具体的民生/政策问题：如"新加坡吸烟规定"、"入境违禁品"

2. **模糊不清或过于口语化的查询**（需要重写）：
   - 过于宽泛：如"去哪玩"、"吃什么"、"推荐酒店"
   - 缺乏具体信息：如"便宜的"、"特色"、"交通"
   - 口语化表达：如"想买点东西回去送人"、"晚上有啥好玩的"

重写原则：
- 保持原意不变
- 增加标准的新加坡旅游术语（如结合区域：Orchard, Bugis, Little India 等）
- 匹配元数据中的核心分类（如 Accommodation/Hotels, Food/Dining, Nightlife/Bars）

示例：
- "去哪玩" → "新加坡 必去景点 观光推荐"
- "想买点东西回去送人" → "新加坡 伴手礼 纪念品 购物 (Souvenirs/Shopping)"
- "晚上有啥好玩的" → "新加坡 夜生活 酒吧 俱乐部 (Nightlife/Bars/Clubs)"
- "吃点好的" → "新加坡 推荐餐厅 美食 高级餐饮 (Food/Dining/Splurge)"
- "乌节路酒店推荐" → "乌节路酒店推荐"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
                {"query": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的旅游问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取推荐列表（如酒店、餐厅、景点列表）
   例如：推荐几个乌节路的平价酒店、武吉士有什么好吃的、给我3个小印度的景点

2. 'detail' - 用户想要具体的路线、攻略或某个地点的详细信息
   例如：怎么从机场去市中心、鱼尾狮公园有什么好玩的、退税的步骤是什么

3. 'general' - 其他一般性、文化类或背景介绍问题
   例如：新加坡的历史是什么、当地的餐桌礼仪、天气怎么样

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
                {"query": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        if result in ['list', 'detail', 'general']:
            return result
        else:
            return 'general'

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询 (如酒店、餐厅、景点)

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，指南中暂无相关的推荐信息。"

        # 提取目标名称 (优先提取 Item_Name，若无则提取 Title/Sub_Section)
        items = []
        for doc in context_docs:
            item_name = doc.metadata.get('Item_Name')
            if not item_name:
                item_name = doc.metadata.get('Sub_Section')
            if not item_name:
                item_name = doc.metadata.get('Title', '未知地点')

            # 去重
            if item_name not in items:
                items.append(item_name)

        # 构建简洁的列表回答
        if len(items) == 1:
            return f"为您推荐：{items[0]}"
        elif len(items) <= 5:
            return f"为您精选以下推荐：\n" + "\n".join([f"{i + 1}. {name}" for i, name in enumerate(items)])
        else:
            return f"为您精选以下热门推荐：\n" + "\n".join([f"{i + 1}. {name}" for i, name in enumerate(
                items[:5])]) + f"\n\n此外，还有其他 {len(items) - 5} 个选择可供探索。"

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

        chain = (
                {"question": RunnablePassthrough(), "context": lambda _: context}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """生成详细攻略回答 - 流式输出"""
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的新加坡旅行规划师。请根据指南信息，为用户提供结构化的旅行攻略。

用户问题: {question}

相关指南信息:
{context}

请灵活组织回答，建议包含以下部分：
## 📍 地点/主题概述
## 🚶‍♂️ 交通指南
## 🌟 推荐体验
## 💡 实用贴士

回答:""")

        chain = (
                {"question": RunnablePassthrough(), "context": lambda _: context}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 2500) -> str:
        """
        构建上下文字符串，适配新加坡旅游指南元数据

        Args:
            docs: 文档列表
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关旅游指南信息。"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            # 获取元数据
            meta = doc.metadata
            category = meta.get('category', '一般信息')
            title = meta.get('Title', '')
            section = meta.get('Section', '')
            sub_section = meta.get('Sub_Section', '')
            item_name = meta.get('Item_Name', '')

            # 组合路径，类似面包屑导航，帮LLM理解层级 (例如: 餐饮 > 预算 > 店名)
            hierarchy_parts = [p for p in [title, section, sub_section, item_name] if p]
            hierarchy = " > ".join(hierarchy_parts)

            metadata_info = f"【指南 {i}】 {hierarchy} | 分类: {category}"

            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n" + "=" * 50 + "\n".join(context_parts)