# ============================================================
# Module: Mood Pool (mood_pool.py)
# 模块：装饰心情池
#
# Stores a pool of decorative mood entries per character.
# 为每个角色存储一组装饰性心情条目。
#
# A "mood" here is a soft, presentational layer — not a PANAS
# measurement. Each entry has a word (like "✨ 月光下的安静")
# and a feeling_word that maps to the emotion lexicon.
# 这里的"心情"是展示层的柔性描述，不是 PANAS 测量。
# 每条条目有一个展示词（如"✨ 月光下的安静"）和一个指向
# 情绪词典的 feeling_word。
#
# Daily selection is deterministic via md5(character + date).
# 每日选择通过 md5(character + date) 确定，保持一天内稳定。
# ============================================================

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from datetime import date
from typing import Any

from emotion_lexicon import lookup

logger = logging.getLogger("ombre_brain.mood_pool")

# ------------------------------------------------------------
# Default pool entries for character='default'
# 默认角色的心情池条目（至少 15 条，覆盖不同象限）
# ------------------------------------------------------------
_DEFAULT_ENTRIES: list[dict[str, str]] = [
    # Q1: 正效价 + 高唤醒
    {"word": "✨ 今天有点小雀跃",        "feeling_word": "雀跃",   "kind": "sparkle"},
    {"word": "🌟 阳光打在脸上的瞬间",    "feeling_word": "喜悦",   "kind": "sparkle"},
    {"word": "🎵 脑子里有首歌出不去",    "feeling_word": "兴奋",   "kind": "sparkle"},
    {"word": "💫 有想聊的冲动",           "feeling_word": "期待",   "kind": "sparkle"},
    {"word": "🌈 莫名其妙地感激",        "feeling_word": "感动",   "kind": "sparkle"},
    # Q2: 负效价 + 高唤醒
    {"word": "⚡ 心里有点燥",            "feeling_word": "烦躁",   "kind": "cloud"},
    {"word": "🌩 有什么东西卡住了",      "feeling_word": "焦虑",   "kind": "cloud"},
    {"word": "🫧 说不清楚的担心",        "feeling_word": "担忧",   "kind": "cloud"},
    # Q3: 负效价 + 低唤醒
    {"word": "🌧 今天有点沉",            "feeling_word": "忧郁",   "kind": "rain"},
    {"word": "🍂 像秋天的叶子，轻飘",    "feeling_word": "失落",   "kind": "rain"},
    {"word": "🌑 有点不想说话",          "feeling_word": "疲惫",   "kind": "rain"},
    {"word": "🪷 安静地想着什么",        "feeling_word": "孤独",   "kind": "rain"},
    # Q4: 正效价 + 低唤醒
    {"word": "🌙 月光下的安静",          "feeling_word": "宁静",   "kind": "moon"},
    {"word": "☕ 喝了口热的，暖着",      "feeling_word": "舒适",   "kind": "moon"},
    {"word": "🌿 像植物一样吸收阳光",    "feeling_word": "放松",   "kind": "moon"},
    {"word": "🍃 今天很轻，也很软",      "feeling_word": "安心",   "kind": "moon"},
    {"word": "✉️ 有点想写信给某人",      "feeling_word": "温暖",   "kind": "sparkle"},
]


class MoodPool:
    """
    Decorative mood pool per character, backed by SQLite.
    基于 SQLite 的按角色装饰心情池。
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Ombre Brain config dict.
        """
        buckets_dir = config.get("buckets_dir", "./buckets")
        os.makedirs(buckets_dir, exist_ok=True)
        self.db_path = os.path.join(buckets_dir, "mood_pool.db")
        self._init_db()

    # ----------------------------------------------------------
    # DB init & seeding
    # ----------------------------------------------------------

    def _init_db(self) -> None:
        """Create mood_pool table and seed default entries if empty."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mood_pool (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character TEXT NOT NULL DEFAULT 'default',
                kind TEXT NOT NULL DEFAULT 'sparkle',
                word TEXT NOT NULL,
                feeling_word TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()

        # Seed defaults for 'default' character if table is empty
        count = conn.execute(
            "SELECT COUNT(*) FROM mood_pool WHERE character = 'default'"
        ).fetchone()[0]
        if count == 0:
            for i, entry in enumerate(_DEFAULT_ENTRIES):
                conn.execute(
                    """
                    INSERT INTO mood_pool (character, kind, word, feeling_word, sort_order)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("default", entry["kind"], entry["word"], entry["feeling_word"], i),
                )
            conn.commit()
            logger.info(f"Seeded {len(_DEFAULT_ENTRIES)} default mood pool entries")

        conn.close()

    # ----------------------------------------------------------
    # Core API
    # ----------------------------------------------------------

    def today_mood(self, character: str = "default") -> dict | None:
        """
        Get today's deterministic mood entry for a character.
        基于 md5(character:date) 确定性地选取今日心情。

        The selection is stable within a calendar day but changes each day.
        同一天内稳定，每天自动换。

        Args:
            character: Character name.

        Returns:
            Mood dict with {id, character, kind, word, feeling, valence, arousal}
            or None if pool is empty.
        """
        entries = self.list_pool(character)
        if not entries:
            return None

        # Deterministic selection: md5 of "character:YYYY-MM-DD"
        key = f"{character}:{date.today().isoformat()}"
        digest = hashlib.md5(key.encode()).hexdigest()
        index = int(digest[:8], 16) % len(entries)
        entry = entries[index]

        # Resolve V/A from emotion lexicon
        coords = lookup(entry["feeling_word"])
        valence = coords[0] if coords else 0.0
        arousal = coords[1] if coords else 0.3

        return {
            "id": entry["id"],
            "character": entry["character"],
            "kind": entry["kind"],
            "word": entry["word"],
            "feeling": entry["feeling_word"],
            "valence": valence,
            "arousal": arousal,
        }

    def list_pool(self, character: str = "default") -> list[dict]:
        """
        List all mood pool entries for a character.
        列出角色的所有心情池条目。

        Args:
            character: Character name.

        Returns:
            List of entry dicts {id, character, kind, word, feeling_word, sort_order}.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """
            SELECT id, character, kind, word, feeling_word, sort_order
            FROM mood_pool
            WHERE character = ?
            ORDER BY sort_order ASC, id ASC
            """,
            (character,),
        ).fetchall()
        conn.close()
        return [
            {
                "id": row[0],
                "character": row[1],
                "kind": row[2],
                "word": row[3],
                "feeling_word": row[4],
                "sort_order": row[5],
            }
            for row in rows
        ]

    def add_entry(
        self,
        character: str,
        word: str,
        feeling_word: str,
        kind: str = "sparkle",
    ) -> int:
        """
        Add a new mood pool entry.
        添加新的心情池条目。

        Args:
            character: Character name.
            word: Display word / phrase (e.g. "✨ 今天有点小雀跃").
            feeling_word: Emotion word mapped to lexicon.
            kind: Visual kind tag (sparkle / cloud / rain / moon).

        Returns:
            New entry id.
        """
        conn = sqlite3.connect(self.db_path)
        # Sort order = max existing + 1
        max_order = conn.execute(
            "SELECT MAX(sort_order) FROM mood_pool WHERE character = ?", (character,)
        ).fetchone()[0]
        sort_order = (max_order or 0) + 1
        cursor = conn.execute(
            """
            INSERT INTO mood_pool (character, kind, word, feeling_word, sort_order)
            VALUES (?, ?, ?, ?, ?)
            """,
            (character, kind, word, feeling_word, sort_order),
        )
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()
        return new_id

    def delete_entry(self, entry_id: int) -> bool:
        """
        Delete a mood pool entry by id.
        按 id 删除心情池条目。

        Args:
            entry_id: Entry id to delete.

        Returns:
            True if deleted, False if not found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "DELETE FROM mood_pool WHERE id = ?", (entry_id,)
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    def list_characters(self) -> list[str]:
        """
        List all characters that have pool entries.
        列出所有有心情池条目的角色名。

        Returns:
            List of character name strings.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT DISTINCT character FROM mood_pool ORDER BY character"
        ).fetchall()
        conn.close()
        return [row[0] for row in rows]
