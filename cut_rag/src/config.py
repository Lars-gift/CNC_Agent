import logging

# 声明模块专用logger（使用全局配置）
logger = logging.getLogger(__name__)

run_chroma = "chroma run --path ./chroma_data --host localhost --port 8000"