# ============================================================
# Module: Emotion Lexicon (emotion_lexicon.py)
# 模块：情绪词典
#
# Closed-set Chinese emotion word dictionary with Russell
# Valence/Arousal (V/A) coordinates.
# 封闭集中文情绪词典，包含 Russell 环状模型的效价/唤醒度坐标。
#
# V = -1.0 (very negative) → +1.0 (very positive)
# A =  0.0 (very calm)    →  1.0 (very intense)
#
# Includes:
# - ~100 words covering all 4 Russell quadrants
# - Exact lookup
# - Multi-layer fallback (exact → backup list → substring 2-4 chars)
# - V/A fusion formula for blending dict coords with AI coords
# - Direction-filtered candidate list for embedding fallback
# ============================================================

from __future__ import annotations

# ------------------------------------------------------------
# 核心词典：~100 个中文情绪词，覆盖 Russell 四象限
# Q1 正效价 + 高唤醒  (右上)
# Q2 负效价 + 高唤醒  (左上)
# Q3 负效价 + 低唤醒  (左下)
# Q4 正效价 + 低唤醒  (右下)
# ------------------------------------------------------------
_LEXICON: dict[str, tuple[float, float]] = {
    # ======== Q1: 正效价 + 高唤醒 ========
    "兴奋":     (0.80, 0.85),
    "激动":     (0.75, 0.88),
    "热情":     (0.70, 0.80),
    "开心":     (0.75, 0.70),
    "快乐":     (0.80, 0.65),
    "喜悦":     (0.82, 0.72),
    "欣喜":     (0.78, 0.78),
    "兴高采烈": (0.85, 0.90),
    "雀跃":     (0.78, 0.82),
    "喜欢":     (0.65, 0.60),
    "爱":       (0.80, 0.70),
    "钦佩":     (0.60, 0.65),
    "崇拜":     (0.65, 0.75),
    "惊喜":     (0.65, 0.80),
    "感动":     (0.70, 0.68),
    "活跃":     (0.60, 0.78),
    "期待":     (0.55, 0.72),
    "渴望":     (0.45, 0.78),
    "希望":     (0.60, 0.55),
    "振奋":     (0.72, 0.80),
    "自豪":     (0.75, 0.65),
    "满足":     (0.70, 0.45),

    # ======== Q2: 负效价 + 高唤醒 ========
    "愤怒":     (-0.82, 0.90),
    "暴怒":     (-0.90, 0.95),
    "恐惧":     (-0.78, 0.85),
    "惊恐":     (-0.72, 0.88),
    "焦虑":     (-0.60, 0.78),
    "紧张":     (-0.40, 0.75),
    "担忧":     (-0.50, 0.65),
    "惊吓":     (-0.65, 0.82),
    "厌恶":     (-0.78, 0.72),
    "愤慨":     (-0.75, 0.80),
    "嫉妒":     (-0.60, 0.78),
    "仇恨":     (-0.88, 0.85),
    "憎恶":     (-0.82, 0.80),
    "心疼":     (-0.40, 0.60),  # PDF specified: v=-0.4, a=0.6
    "戒备":     (-0.15, 0.70),  # PDF specified: v=-0.15, a=0.7
    "占有欲":   (-0.30, 0.75),  # PDF specified: v=-0.3, a=0.75
    "慌张":     (-0.55, 0.80),
    "惶恐":     (-0.65, 0.82),
    "不安":     (-0.55, 0.72),
    "急躁":     (-0.50, 0.78),
    "烦躁":     (-0.52, 0.72),

    # ======== Q3: 负效价 + 低唤醒 ========
    "悲伤":     (-0.78, 0.45),
    "难过":     (-0.65, 0.42),
    "忧郁":     (-0.60, 0.35),
    "郁闷":     (-0.55, 0.40),
    "失落":     (-0.58, 0.38),
    "失望":     (-0.62, 0.42),
    "沮丧":     (-0.65, 0.45),
    "无助":     (-0.70, 0.30),
    "绝望":     (-0.88, 0.35),
    "孤独":     (-0.62, 0.30),
    "寂寞":     (-0.55, 0.28),
    "空虚":     (-0.50, 0.25),
    "疲惫":     (-0.40, 0.20),
    "冷漠":     (-0.30, 0.18),
    "麻木":     (-0.35, 0.15),
    "委屈":     (-0.60, 0.48),
    "内疚":     (-0.58, 0.50),
    "羞愧":     (-0.62, 0.55),
    "后悔":     (-0.55, 0.45),
    "遗憾":     (-0.48, 0.40),
    "哀愁":     (-0.65, 0.30),
    "忧伤":     (-0.68, 0.35),

    # ======== Q4: 正效价 + 低唤醒 ========
    "安心":     (0.50, 0.30),    # PDF specified: v=0.5, a=0.3
    "平静":     (0.05, 0.18),    # PDF specified: v=0.05, a=0.18 (close to neutral)
    "羞涩":     (0.025, 0.52),   # PDF specified: v=0.025, a=0.52
    "放松":     (0.55, 0.22),
    "舒适":     (0.60, 0.25),
    "温暖":     (0.65, 0.35),
    "感激":     (0.68, 0.42),
    "幸福":     (0.82, 0.38),
    "满意":     (0.65, 0.28),
    "宁静":     (0.40, 0.15),
    "祥和":     (0.45, 0.12),
    "怀念":     (0.20, 0.30),
    "惬意":     (0.62, 0.25),
    "安慰":     (0.52, 0.32),
    "欣慰":     (0.60, 0.38),
    "轻松":     (0.58, 0.28),
    "释然":     (0.48, 0.22),
    "从容":     (0.42, 0.20),
    "慵懒":     (0.25, 0.12),
    "温柔":     (0.60, 0.30),
    "体贴":     (0.58, 0.32),
    "呵护":     (0.55, 0.38),

    # ======== 接近中性 ========
    "淡然":     (0.08, 0.15),
    "茫然":     (-0.15, 0.25),
    "迷茫":     (-0.20, 0.30),
    "困惑":     (-0.18, 0.35),
    "好奇":     (0.30, 0.55),
    "惊讶":     (0.10, 0.75),
    "尴尬":     (-0.25, 0.52),
    "害羞":     (0.10, 0.50),
    "羡慕":     (0.20, 0.55),
    "无聊":     (-0.22, 0.15),
}


def lookup(word: str) -> tuple[float, float] | None:
    """
    Exact lookup of a word in the emotion lexicon.
    精确查找情绪词。

    Args:
        word: Chinese emotion word to look up.

    Returns:
        (valence, arousal) tuple if found, else None.
        valence: -1.0 (very negative) → +1.0 (very positive)
        arousal:  0.0 (very calm)    →  1.0 (very intense)
    """
    return _LEXICON.get(word)


def lookup_with_fallback(
    primary: str,
    backup_words: list[str] | None = None,
) -> tuple[tuple[float, float] | None, str]:
    """
    Multi-layer fallback lookup.
    多层回退查找。

    Layer 1: exact match on primary word
    Layer 2: exact match on each backup word (in order)
    Layer 3: substring match (2-4 chars) from primary word substrings

    Args:
        primary: Primary emotion word from LLM.
        backup_words: Optional list of backup words from LLM.

    Returns:
        ((valence, arousal), source_label) — source_label indicates which
        layer matched (exact / backup / substring / None).
    """
    # Layer 1: exact match
    result = _LEXICON.get(primary)
    if result is not None:
        return result, "exact"

    # Layer 2: backup words
    for bw in (backup_words or []):
        result = _LEXICON.get(bw)
        if result is not None:
            return result, "backup"

    # Layer 3: substring (2-4 chars) from primary word
    word = primary
    for length in (4, 3, 2):
        for start in range(len(word) - length + 1):
            substr = word[start: start + length]
            result = _LEXICON.get(substr)
            if result is not None:
                return result, "substring"

    return None, "none"


def fuse_va(
    dict_v: float,
    dict_a: float,
    ai_v_01: float,
    ai_a: float,
) -> tuple[float, float]:
    """
    Fuse dictionary V/A with AI-predicted V/A.
    融合词典 V/A 与 AI 预测的 V/A。

    Formula (PDF spec):
        final_V = 0.7 × dict_V + 0.3 × ai_V_normalized
        final_A = 0.7 × dict_A + 0.3 × ai_A

    where ai_V_01 is in [0, 1] range (AI output), converted to [-1, 1]:
        ai_V_normalized = ai_V_01 * 2 - 1

    Args:
        dict_v: Lexicon valence (-1 to +1).
        dict_a: Lexicon arousal (0 to 1).
        ai_v_01: AI-predicted valence in 0-1 range.
        ai_a: AI-predicted arousal in 0-1 range.

    Returns:
        (fused_valence, fused_arousal)
    """
    ai_v_norm = ai_v_01 * 2.0 - 1.0  # convert 0..1 → -1..1
    final_v = 0.7 * dict_v + 0.3 * ai_v_norm
    final_a = 0.7 * dict_a + 0.3 * ai_a
    # Clamp to valid ranges
    final_v = max(-1.0, min(1.0, final_v))
    final_a = max(0.0, min(1.0, final_a))
    return round(final_v, 4), round(final_a, 4)


def get_candidates_by_direction(
    valence_positive: bool | None = None,
    arousal_high: bool | None = None,
) -> list[str]:
    """
    Return candidate words filtered by Russell quadrant direction.
    按 Russell 象限方向返回候选词列表（用于 embedding fallback 候选集筛选）。

    Args:
        valence_positive: True → positive valence; False → negative; None → both.
        arousal_high:     True → arousal >= 0.5;   False → arousal < 0.5; None → both.

    Returns:
        List of matching Chinese emotion words.
    """
    results = []
    for word, (v, a) in _LEXICON.items():
        v_ok = (
            valence_positive is None
            or (valence_positive and v >= 0)
            or (not valence_positive and v < 0)
        )
        a_ok = (
            arousal_high is None
            or (arousal_high and a >= 0.5)
            or (not arousal_high and a < 0.5)
        )
        if v_ok and a_ok:
            results.append(word)
    return results


def all_words() -> list[str]:
    """Return all words in the lexicon."""
    return list(_LEXICON.keys())
