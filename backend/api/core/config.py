import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,https://t-autonlp.vercel.app",
    ).split(",")


settings = Settings()
