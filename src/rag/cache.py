"""Disposable local caches for the lexical index and its ranked results."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


CACHE_VERSION = 1


def index_signature(path: Path) -> str:
    """Identify a chunks generation without reading the entire corpus."""
    stat = path.stat()
    return (
        f"{CACHE_VERSION}:{path.resolve()}:{stat.st_dev}:{stat.st_ino}:"
        f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"
    )


class QueryCache:
    """Store at most 1000 ranked queries, shared across CLI processes."""

    def __init__(self, path: Path, signature: str) -> None:
        """Bind cached chunk positions to one version of the index."""
        self.path = path
        self.signature = signature

    def _connect(self) -> sqlite3.Connection:
        """Open the database and discard results from older index versions."""
        connection = sqlite3.connect(self.path, timeout=1)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS queries ("
                "signature TEXT, query TEXT, k INTEGER, positions TEXT, "
                "used INTEGER, PRIMARY KEY (signature, query, k))"
            )
            connection.execute(
                "DELETE FROM queries WHERE signature != ?", (self.signature,),
            )
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def get(self, query: str, k: int, chunk_count: int) -> list[int] | None:
        """Read positions, treating unavailable or corrupt caches as misses."""
        try:
            with closing(self._connect()) as connection, connection:
                row = connection.execute(
                    "SELECT positions FROM queries "
                    "WHERE signature = ? AND query = ? AND k = ?",
                    (self.signature, query, k),
                ).fetchone()
                if row is None:
                    return None
                positions = json.loads(row[0])
                if (not isinstance(positions, list) or len(positions) > k
                        or any(type(i) is not int or not 0 <= i < chunk_count
                               for i in positions)):
                    return None
                connection.execute(
                    "UPDATE queries SET used = "
                    "(SELECT COALESCE(MAX(used), 0) + 1 FROM queries) "
                    "WHERE signature = ? AND query = ? AND k = ?",
                    (self.signature, query, k),
                )
                return positions
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return None

    def put(self, query: str, k: int, positions: list[int]) -> None:
        """Save a result and evict the least recently used entries."""
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "INSERT OR REPLACE INTO queries VALUES (?, ?, ?, ?, "
                    "(SELECT COALESCE(MAX(used), 0) + 1 FROM queries))",
                    (self.signature, query, k, json.dumps(positions)),
                )
                connection.execute(
                    "DELETE FROM queries WHERE rowid NOT IN "
                    "(SELECT rowid FROM queries ORDER BY used DESC LIMIT 1000)"
                )
        except (OSError, sqlite3.Error):
            pass
