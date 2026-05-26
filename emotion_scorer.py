# ============================================================
# Module: Emotion Scorer (emotion_scorer.py)
# 模块：PANAS 情绪评分引擎
#
# Scores conversation text for Positive Affect (PA) and
# Negative Affect (NA) using an LLM, then resolves emotion
# word V/A coordinates via a 5-layer matching pipeline.
# 通过 LLM 对对话文本打 PA/NA 分，并用 5 层匹配管线解析情绪词 V/A 坐标。
#
# Core responsibilities:
# 核心职责：
#   - Call scoring LLM (fire-and-forget from server.py)
#   - 5-layer word matching: exact → backup → substring → embedding → free_form
#   - Persist events to SQLite (buckets_dir/mood_events.db)
#   - Expose PANAS today summary with time-decay weighting
#   - Expose recent high-arousal words and recent events list
# ============================================================

from __future__ import annotations

import json
import math
import os
import sqlite3
import logging
from datetime import datetime, date, timezone
from typing import Any

import httpx

from emotion_lexicon import lookup_with_fallback, fuse_va, get_candidates_by_direction

logger = logging.getLogger("ombre_brain.emotion")

# ------------------------------------------------------------
# LLM scoring prompt
# 评分 LLM 提示词
# ------------------------------------------------------------
_SCORE_PROMPT = """你是一个情绪分析助手，负责对一段对话文本进行 PANAS 情绪评估。

PANAS 说明：
- PA (Positive Affect)：正向情感，如兴奋、热情、快乐。范围 [-1, 1]，正值表示 PA 增加。
- NA (Negative Affect)：负向情感，如焦虑、难过、烦躁。范围 [-1, 1]，正值表示 NA 增加。

注意：
- 冷场就是冷场（PA低NA低），敷衍就是敷衍（PA低），不允许每次都评高分
- 中性对话应该评接近 0 的分数
- 只有明显的情绪变化才给较高分

请对以下对话文本打分，并以 JSON 格式输出：

```json
{
  "pa_delta": 0.0,
  "na_delta": 0.0,
  "valence": 0.5,
  "arousal": 0.3,
  "word": "主要情绪词（中文单词）",
  "backup": ["备选词1", "备选词2"],
  "reason": "一句话说明打分依据"
}
```

字段说明：
- pa_delta: 正向情感变化，-1 到 1
- na_delta: 负向情感变化，-1 到 1
- valence: 情绪效价，0（极负）到 1（极正），0.5 为中性
- arousal: 唤醒度，0（极平静）到 1（极强烈）
- word: 最能概括这段文字情绪的中文单词
- backup: 2-3 个备选情绪词
- reason: 简短说明

对话文本：
"""


class EmotionScorer:
    """
    PANAS emotion scoring engine with SQLite persistence.
    带 SQLite 持久化的 PANAS 情绪评分引擎。
    """

    def __init__(self, config: dict, embedding_engine=None):
        """
        Args:
            config: Ombre Brain config dict (uses config["dehydration"] for LLM).
            embedding_engine: Optional EmbeddingEngine instance for layer-4 matching.
        """
        self.config = config
        self.embedding_engine = embedding_engine

        dehy_cfg = config.get("dehydration", {})
        self.api_key = dehy_cfg.get("api_key", "")
        self.base_url = dehy_cfg.get("base_url", "")
        self.model = dehy_cfg.get("model", "deepseek-chat")

        # SQLite path: buckets_dir/mood_events.db
        buckets_dir = config.get("buckets_dir", "./buckets")
        os.makedirs(buckets_dir, exist_ok=True)
        self.db_path = os.path.join(buckets_dir, "mood_events.db")
        self._init_db()

    # ----------------------------------------------------------
    # DB init
    # ----------------------------------------------------------

    def _init_db(self) -> None:
        """Create mood_events table if not exists."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mood_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                date_str TEXT NOT NULL,
                pa_delta REAL NOT NULL DEFAULT 0,
                na_delta REAL NOT NULL DEFAULT 0,
                valence REAL NOT NULL DEFAULT 0,
                arousal REAL NOT NULL DEFAULT 0,
                word TEXT NOT NULL DEFAULT '',
                match_source TEXT NOT NULL DEFAULT 'none',
                reason TEXT NOT NULL DEFAULT '',
                weight REAL NOT NULL DEFAULT 1.0
            )
        """)
        conn.commit()
        conn.close()

    # ----------------------------------------------------------
    # Core scoring pipeline
    # ----------------------------------------------------------

    async def score_emotion(self, conversation_text: str) -> dict:
        """
        Score a conversation text for emotion. Async, fire-and-forget safe.
        对对话文本进行情绪评分，异步，适合 fire-and-forget 调用。

        Steps:
        1. Call scoring LLM
        2. 5-layer word matching
        3. Fuse V/A
        4. Persist to SQLite

        Returns:
            dict with all event fields, or {"error": ...} on failure.
        """
        # Step 1: LLM scoring
        try:
            llm_result = await self._call_scoring_llm(conversation_text)
        except Exception as e:
            logger.warning(f"Emotion scoring LLM call failed: {e}")
            return {"error": str(e)}

        pa_delta = float(llm_result.get("pa_delta", 0.0))
        na_delta = float(llm_result.get("na_delta", 0.0))
        ai_valence_01 = float(llm_result.get("valence", 0.5))  # 0-1 range
        ai_arousal = float(llm_result.get("arousal", 0.3))
        primary_word = str(llm_result.get("word", ""))
        backup_words = llm_result.get("backup", [])
        if not isinstance(backup_words, list):
            backup_words = []
        reason = str(llm_result.get("reason", ""))

        # Step 2: 5-layer word matching
        final_v, final_a, match_source, matched_word = await self._match_word(
            primary_word, backup_words, ai_valence_01, ai_arousal
        )

        # Step 3: Persist event
        event = {
            "pa_delta": round(pa_delta, 4),
            "na_delta": round(na_delta, 4),
            "valence": round(final_v, 4),
            "arousal": round(final_a, 4),
            "word": matched_word or primary_word,
            "match_source": match_source,
            "reason": reason,
            "weight": 1.0,
        }
        self._persist_event(event)
        logger.info(
            f"Emotion scored: pa={pa_delta:.2f} na={na_delta:.2f} "
            f"word={event['word']} source={match_source}"
        )
        return event

    # ----------------------------------------------------------
    # LLM call
    # ----------------------------------------------------------

    async def _call_scoring_llm(self, text: str) -> dict:
        """
        Call the scoring LLM and parse JSON response.
        调用评分 LLM 并解析 JSON 响应。
        """
        if not self.api_key:
            raise ValueError("No API key configured for emotion scoring")

        prompt = _SCORE_PROMPT + text[:3000]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 512,
        }

        # Build API URL
        base = self.base_url.rstrip("/")
        if not base:
            base = "https://api.deepseek.com"
        url = f"{base}/chat/completions"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]

        # Extract JSON from response (may be wrapped in markdown code block)
        content = content.strip()
        if "```" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Attempt to extract embedded JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            raise ValueError(f"Could not parse LLM JSON response: {content[:200]}")

    # ----------------------------------------------------------
    # 5-layer word matching
    # ----------------------------------------------------------

    async def _match_word(
        self,
        primary: str,
        backup: list[str],
        ai_valence_01: float,
        ai_arousal: float,
    ) -> tuple[float, float, str, str]:
        """
        5-layer matching pipeline to resolve final V/A coordinates.
        5 层匹配管线，解析最终 V/A 坐标。

        Layer 1: exact dict lookup on primary word
        Layer 2: exact dict lookup on backup words
        Layer 3: substring (2-4 chars) of primary word
        Layer 4: embedding similarity within direction-filtered candidates
        Layer 5: free_form — use AI V/A directly (no dict match)

        Returns:
            (final_valence, final_arousal, match_source, matched_word)
        """
        # Layers 1-3 via lookup_with_fallback
        coords, source = lookup_with_fallback(primary, backup)
        if coords is not None:
            dict_v, dict_a = coords
            fused_v, fused_a = fuse_va(dict_v, dict_a, ai_valence_01, ai_arousal)
            return fused_v, fused_a, source, primary

        # Layer 4: embedding similarity
        if self.embedding_engine and self.embedding_engine.enabled:
            try:
                result = await self._embedding_match(primary, ai_valence_01, ai_arousal)
                if result is not None:
                    fused_v, fused_a, emb_word = result
                    return fused_v, fused_a, "embedding", emb_word
            except Exception as e:
                logger.warning(f"Embedding match failed: {e}")

        # Layer 5: free_form — convert AI valence from 0-1 to -1..1 directly
        ai_v_norm = ai_valence_01 * 2.0 - 1.0
        return round(ai_v_norm, 4), round(ai_arousal, 4), "free_form", primary

    async def _embedding_match(
        self,
        word: str,
        ai_valence_01: float,
        ai_arousal: float,
    ):
        """
        Layer 4: use embedding similarity to find best matching lexicon word.
        第 4 层：用 embedding 相似度在候选词集合里找最匹配的词典词。

        Returns:
            (fused_v, fused_a, matched_word) or None if no good match.
        """
        # Build direction-filtered candidate list
        valence_pos = ai_valence_01 >= 0.5
        arousal_hi = ai_arousal >= 0.5
        candidates = get_candidates_by_direction(valence_pos, arousal_hi)
        if not candidates:
            candidates = get_candidates_by_direction()  # fallback: all words

        # Generate embedding for query word
        query_emb = await self.embedding_engine._generate_embedding(word)
        if not query_emb:
            return None

        best_sim = -1.0
        best_word = None
        for cand in candidates:
            cand_emb = await self.embedding_engine._generate_embedding(cand)
            if not cand_emb:
                continue
            sim = self._cosine_similarity(query_emb, cand_emb)
            if sim > best_sim:
                best_sim = sim
                best_word = cand

        if best_word is None or best_sim < 0.3:
            return None

        from emotion_lexicon import lookup
        coords = lookup(best_word)
        if coords is None:
            return None

        dict_v, dict_a = coords
        fused_v, fused_a = fuse_va(dict_v, dict_a, ai_valence_01, ai_arousal)
        return fused_v, fused_a, best_word

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def _persist_event(self, event: dict) -> None:
        """Write a mood event row to SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        date_str = date.today().isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO mood_events
                (created_at, date_str, pa_delta, na_delta, valence, arousal,
                 word, match_source, reason, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                date_str,
                event["pa_delta"],
                event["na_delta"],
                event["valence"],
                event["arousal"],
                event["word"],
                event["match_source"],
                event["reason"],
                event.get("weight", 1.0),
            ),
        )
        conn.commit()
        conn.close()

    # ----------------------------------------------------------
    # Query helpers
    # ----------------------------------------------------------

    def get_panas_today(self) -> dict:
        """
        Get today's cumulative PANAS with time-decay weighting.
        获取今日 PANAS 累积值（带时间衰减）。

        Decay formula: w = exp(-0.1 * age_hours)
        Events older than 24 hours are excluded.

        Returns:
            {"pa": float, "na": float}
        """
        now = datetime.now(timezone.utc)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT created_at, pa_delta, na_delta FROM mood_events ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        conn.close()

        pa = 0.0
        na = 0.0
        for created_at_str, pa_delta, na_delta in rows:
            try:
                created_at = datetime.fromisoformat(created_at_str)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                age_hours = (now - created_at).total_seconds() / 3600.0
                if age_hours > 24:
                    break
                w = math.exp(-0.1 * age_hours)
                pa += pa_delta * w
                na += na_delta * w
            except Exception:
                continue

        return {"pa": round(pa, 4), "na": round(na, 4)}

    def get_recent_high_arousal_words(self, limit: int = 5) -> list[str]:
        """
        Get recent emotion words with arousal > 0.6.
        获取最近的高唤醒情绪词（arousal > 0.6）。

        Args:
            limit: Maximum number of words to return.

        Returns:
            List of emotion word strings.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """
            SELECT word FROM mood_events
            WHERE arousal > 0.6
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [row[0] for row in rows if row[0]]

    def get_recent_events(self, limit: int = 20) -> list[dict]:
        """
        Get recent mood events for dashboard display.
        获取最近的情绪事件（仪表板用）。

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of event dicts.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """
            SELECT id, created_at, date_str, pa_delta, na_delta, valence, arousal,
                   word, match_source, reason
            FROM mood_events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {
                "id": row[0],
                "created_at": row[1],
                "date_str": row[2],
                "pa_delta": row[3],
                "na_delta": row[4],
                "valence": row[5],
                "arousal": row[6],
                "word": row[7],
                "match_source": row[8],
                "reason": row[9],
            }
            for row in rows
        ]
