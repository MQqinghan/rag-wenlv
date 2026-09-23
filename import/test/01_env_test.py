import os
from dotenv import load_dotenv

load_dotenv(
    override=True
)
print(os.environ.get("BGE_M3_PATH"))