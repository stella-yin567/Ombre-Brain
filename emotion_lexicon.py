# ============================================================
# Module: Emotion Lexicon (emotion_lexicon.py)
# 模块：情绪词典
#
# 封闭集中文情绪词 V/A 坐标词典，基于 Russell 环形情绪空间。
# Closed-set Chinese emotion word dictionary with Russell V/A coordinates.
#
# V (valence):  -1.0 = 非常负面  →  +1.0 = 非常正面
# A (arousal):   0.0 = 非常平静  →   1.0 = 非常激烈
#
# Depended on by: emotion_scorer.py, mood_pool.py, server.py
# ============================================================

from __future__ import annotations

import math
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# 核心词典：~100 条中文情绪词，覆盖 Russell 空间四个象限
# Core lexicon: ~100 Chinese emotion words covering all 4 Russell quadrants
#
# Quadrant layout:
#   Q1 (+V, +A): 高兴/兴奋     High pleasure + High arousal
#   Q2 (-V, +A): 愤怒/恐惧     Low pleasure + High arousal
#   Q3 (-V, -A): 悲伤/抑郁     Low pleasure + Low arousal
#   Q4 (+V, -A): 平静/满足     High pleasure + Low arousal
# ──────────────────────────────────────────────────────────────────────

_LEXICON: dict[str, tuple[float, float]] = {
    # ── Q1: 正面 + 高唤醒 ─────────────────────────────────────────────
    "兴奋":    ( 0.85,  0.90),
    "激动":    ( 0.75,  0.88),
    "狂喜":    ( 0.95,  0.95),
    "欣喜":    ( 0.80,  0.78),
    "快乐":    ( 0.80,  0.72),
    "高兴":    ( 0.75,  0.65),
    "开心":    ( 0.78,  0.70),
    "喜悦":    ( 0.82,  0.68),
    "热情":    ( 0.70,  0.82),
    "期待":    ( 0.60,  0.75),
    "渴望":    ( 0.55,  0.80),
    "向往":    ( 0.62,  0.72),
    "活力":    ( 0.72,  0.85),
    "振奋":    ( 0.68,  0.80),
    "愉悦":    ( 0.78,  0.62),
    "惊喜":    ( 0.70,  0.82),
    "雀跃":    ( 0.80,  0.85),
    "欢呼":    ( 0.85,  0.88),
    "甜蜜":    ( 0.80,  0.55),
    "心动":    ( 0.72,  0.75),

    # ── Q2: 负面 + 高唤醒 ─────────────────────────────────────────────
    "愤怒":    (-0.85,  0.92),
    "暴怒":    (-0.95,  0.97),
    "恐惧":    (-0.80,  0.90),
    "惊恐":    (-0.85,  0.95),
    "焦虑":    (-0.55,  0.80),
    "紧张":    (-0.40,  0.78),
    "不安":    (-0.45,  0.72),
    "担忧":    (-0.50,  0.68),
    "恐慌":    (-0.88,  0.95),
    "嫉妒":    (-0.60,  0.75),
    "嫌弃":    (-0.65,  0.65),
    "厌恶":    (-0.78,  0.72),
    "憎恨":    (-0.90,  0.85),
    "愤恨":    (-0.88,  0.88),
    "仇恨":    (-0.92,  0.80),
    "戒备":    (-0.15,  0.70),   # 规格中指定 v=-0.15, a=0.7
    "占有欲":  (-0.30,  0.75),   # 规格中指定 v=-0.30, a=0.75
    "心疼":    (-0.40,  0.60),   # 规格中指定 v=-0.40, a=0.60
    "痛苦":    (-0.82,  0.78),
    "委屈":    (-0.65,  0.68),
    "沮丧":    (-0.68,  0.62),
    "崩溃":    (-0.90,  0.92),
    "烦躁":    (-0.55,  0.75),
    "挫败":    (-0.62,  0.65),
    "羞耻":    (-0.70,  0.72),
    "震惊":    (-0.20,  0.90),
    "惊讶":    (-0.05,  0.85),

    # ── Q3: 负面 + 低唤醒 ─────────────────────────────────────────────
    "悲伤":    (-0.75,  0.45),
    "忧郁":    (-0.65,  0.30),
    "沉默":    (-0.15,  0.15),
    "孤独":    (-0.70,  0.28),
    "绝望":    (-0.92,  0.40),
    "无助":    (-0.80,  0.35),
    "麻木":    (-0.50,  0.12),
    "失落":    (-0.65,  0.38),
    "疲惫":    (-0.40,  0.20),
    "倦怠":    (-0.45,  0.18),
    "无聊":    (-0.30,  0.22),
    "空洞":    (-0.55,  0.15),
    "哀愁":    (-0.62,  0.32),
    "惋惜":    (-0.42,  0.35),
    "遗憾":    (-0.48,  0.32),
    "后悔":    (-0.60,  0.42),
    "思念":    (-0.20,  0.40),
    "伤心":    (-0.72,  0.48),
    "压抑":    (-0.60,  0.35),
    "沮丧":    (-0.65,  0.45),

    # ── Q4: 正面 + 低唤醒 ─────────────────────────────────────────────
    "平静":    ( 0.05,  0.18),   # 规格中指定，接近中性
    "安心":    ( 0.50,  0.30),   # 规格中指定 v=0.5, a=0.3
    "安慰":    ( 0.48,  0.28),
    "满足":    ( 0.72,  0.28),
    "幸福":    ( 0.88,  0.38),
    "温暖":    ( 0.70,  0.32),
    "宁静":    ( 0.55,  0.12),
    "放松":    ( 0.58,  0.20),
    "轻松":    ( 0.60,  0.25),
    "惬意":    ( 0.65,  0.22),
    "感动":    ( 0.65,  0.50),
    "感激":    ( 0.72,  0.40),
    "信任":    ( 0.65,  0.30),
    "踏实":    ( 0.58,  0.22),
    "欣慰":    ( 0.62,  0.32),
    "喜欢":    ( 0.75,  0.48),
    "爱":      ( 0.82,  0.52),
    "依恋":    ( 0.55,  0.42),
    "珍视":    ( 0.70,  0.35),
    "接受":    ( 0.38,  0.22),
    "释然":    ( 0.50,  0.20),

    # ── 复杂/中性边界情绪 ─────────────────────────────────────────────
    "羞涩":    ( 0.025, 0.52),   # 规格中指定 v=0.025, a=0.52
    "害羞":    ( 0.02,  0.50),
    "好奇":    ( 0.40,  0.62),
    "迷茫":    (-0.25,  0.48),
    "困惑":    (-0.20,  0.52),
    "矛盾":    (-0.10,  0.55),
    "复杂":    (-0.05,  0.45),
    "纠结":    (-0.15,  0.60),
    "犹豫":    (-0.10,  0.50),
    "怀念":    ( 0.15,  0.38),
    "无奈":    (-0.38,  0.35),
    "淡然":    ( 0.10,  0.15),
    "冷静":    ( 0.12,  0.20),
}


def lookup(word: str) -> Optional[dict]:
    """
    精确查找情绪词的 V/A 坐标。
    Exact lookup of V/A coordinates for an emotion word.

    Args:
        word: 情绪词 / Emotion word in Chinese.

    Returns:
        {"v": float, "a": float} if found, else None.
    """
    if word in _LEXICON:
        v, a = _LEXICON[word]
        return {"v": v, "a": a}
    return None


def lookup_with_fallback(word: str) -> Optional[dict]:
    """
    多层 fallback 查找：精确 → 子串（2-4字）→ None。
    Multi-layer fallback lookup: exact → substring (2-4 chars) → None.

    Args:
        word: 待查找的情绪词或包含情绪词的字符串。

    Returns:
        {"v": float, "a": float, "match": str, "source": str} or None.
    """
    # Layer 1: exact match / 精确匹配
    if word in _LEXICON:
        v, a = _LEXICON[word]
        return {"v": v, "a": a, "match": word, "source": "exact"}

    # Layer 2: substring match (2-4 chars) / 子串匹配（2-4字）
    for length in range(min(4, len(word)), 1, -1):
        for start in range(len(word) - length + 1):
            substr = word[start:start + length]
            if substr in _LEXICON:
                v, a = _LEXICON[substr]
                return {"v": v, "a": a, "match": substr, "source": "substring"}

    return None


def fuse_va(dict_v: float, dict_a: float, ai_v: float, ai_a: float) -> tuple[float, float]:
    """
    融合词典 V/A 和 AI 给出的 V/A 值。
    Fuse dictionary V/A with AI-provided V/A using weighted formula.

    Formula: final_V = 0.7 × dict_V + 0.3 × ai_V
             final_A = 0.7 × dict_A + 0.3 × ai_A

    Args:
        dict_v: 词典中的 V 坐标 (-1 to +1)
        dict_a: 词典中的 A 坐标 (0 to 1)
        ai_v:   AI 给出的 V 坐标（已归一化到 -1 to +1）
        ai_a:   AI 给出的 A 坐标 (0 to 1)

    Returns:
        (final_v, final_a) tuple.
    """
    final_v = 0.7 * dict_v + 0.3 * ai_v
    final_a = 0.7 * dict_a + 0.3 * ai_a
    # Clamp to valid range
    final_v = max(-1.0, min(1.0, final_v))
    final_a = max(0.0, min(1.0, final_a))
    return final_v, final_a


def get_candidates_by_direction(target_v: float, target_a: float, tolerance: float = 0.4) -> list[str]:
    """
    按方向过滤候选词，用于 embedding fallback。
    Direction-filtered candidate list for embedding fallback.

    Returns words within ±tolerance in V/A space.
    返回 V/A 空间内方向相近的候选词列表。

    Args:
        target_v:  目标 V 坐标 (-1 to +1)
        target_a:  目标 A 坐标 (0 to 1)
        tolerance: 容差（默认 0.4）

    Returns:
        List of matching emotion words.
    """
    candidates = []
    for word, (v, a) in _LEXICON.items():
        if abs(v - target_v) <= tolerance and abs(a - target_a) <= tolerance:
            candidates.append(word)
    return candidates


def _euclidean_distance(v1: float, a1: float, v2: float, a2: float) -> float:
    """
    计算 Russell 空间内两点的欧氏距离。
    Compute Euclidean distance in Russell space.
    """
    return math.sqrt((v1 - v2) ** 2 + (a1 - a2) ** 2)


def nearest_words(v: float, a: float, top_k: int = 5) -> list[dict]:
    """
    返回 Russell 空间中最接近给定坐标的 K 个情绪词。
    Return K emotion words nearest to the given V/A coordinates.

    Args:
        v:     目标 V 坐标 (-1 to +1)
        a:     目标 A 坐标 (0 to 1)
        top_k: 返回数量

    Returns:
        List of {"word": str, "v": float, "a": float, "dist": float}
    """
    scored = []
    for word, (wv, wa) in _LEXICON.items():
        dist = _euclidean_distance(v, a, wv, wa)
        scored.append({"word": word, "v": wv, "a": wa, "dist": dist})
    scored.sort(key=lambda x: x["dist"])
    return scored[:top_k]


def list_all() -> dict[str, tuple[float, float]]:
    """
    返回完整词典副本。
    Return a copy of the full lexicon.
    """
    return dict(_LEXICON)
