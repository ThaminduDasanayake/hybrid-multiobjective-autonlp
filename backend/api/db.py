import os

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

_MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
_DB_NAME = os.getenv("MONGODB_DB", "tautonlp")

_client: MongoClient | None = None


def get_db():
    """Return the shared database instance for the main (API) process."""
    global _client
    if _client is None:
        _client = MongoClient(_MONGODB_URI, tlsCAFile=certifi.where())
    return _client[_DB_NAME]


def get_mongo_uri() -> str:
    """Return the connection URI (picklable — safe to pass to worker processes)."""
    return _MONGODB_URI


def get_db_name() -> str:
    """Return the database name (picklable — safe to pass to worker processes)."""
    return _DB_NAME
