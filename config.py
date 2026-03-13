"""
RAG系统配置文件
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    chunks_path: str = "./saved_chunks_global.pkl"
    data_path: str = "./wikivoyage_global"
    index_save_path: str = "./vector_index_global"

    # 模型配置
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "kimi-k2-0711-preview"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        """初始化后的处理"""
        env_data_path = os.getenv("TRAVEL_DATA_PATH", "").strip()
        if env_data_path:
            self.data_path = env_data_path
            return

        # Prefer global dataset by default; fall back to SG to keep existing repo usable.
        global_path = Path("./wikivoyage_global")
        sg_path = Path("./wikivoyage_sg")
        if global_path.exists():
            self.data_path = str(global_path)
        elif sg_path.exists():
            self.data_path = str(sg_path)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


# 默认配置实例
DEFAULT_CONFIG = RAGConfig()