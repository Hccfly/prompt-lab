"""极简 .env 加载器（零依赖）。

把项目根目录下的 .env 文件内容注入环境变量，
这样 llm.py 就能通过 os.environ 读取 DEEPSEEK_API_KEY。
"""
import os
from pathlib import Path


def load_dotenv(path=None) -> None:
    path = Path(path or Path(__file__).parent / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()
