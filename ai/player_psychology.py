"""
游戏内玩家心理模型与情感计算系统 - 核心模块

包含五大子系统：
1. emotion_model()        情绪模型：多维度（快乐/愤怒/悲伤/恐惧）+ 强度
2. emotion_recognition()  情绪识别：行为特征 -> 情绪，带置信度
3. behavior_prediction()  行为预测：一阶马尔可夫模型 + 概率分布
4. personality_model()    性格模型：大五人格 + 时间稳定性
5. adaptive_dialogue()    自适应对话：情绪驱动策略 + 反馈学习
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


EMOTIONS = ("joy", "anger", "sadness", "fear")

BIG_FIVE = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

# 行为特征 -> 各情绪的证据权重
BEHAVIOR_EMOTION_FEATURES: Dict[str, Dict[str, float]] = {
    "attack": {"anger": 0.7, "fear": 0.1},
    "destroy": {"anger": 0.8},
    "insult": {"anger": 0.6},
    "explore": {"joy": 0.5},
    "complete_quest": {"joy": 0.8},
    "chat": {"joy": 0.4},
    "help": {"joy": 0.5},
    "die": {"sadness": 0.7, "fear": 0.2},
    "fail": {"sadness": 0.6, "anger": 0.2},
    "lose_item": {"sadness": 0.5, "anger": 0.3},
    "flee": {"fear": 0.7},
    "hide": {"fear": 0.5},
    "avoid": {"fear": 0.4},
}

# 行为特征 -> 大五人格证据
BEHAVIOR_PERSONALITY_FEATURES: Dict[str, Dict[str, float]] = {
    "explore": {"openness": 0.8},
    "complete_quest": {"conscientiousness": 0.8},
    "chat": {"extraversion": 0.8},
    "help": {"agreeableness": 0.8},
    "die": {"neuroticism": 0.6},
    "fail": {"neuroticism": 0.5},
    "flee": {"neuroticism": 0.7},
    "attack": {"agreeableness": 0.2},
}

# 主导情绪 -> 对话策略候选（按优先级）
EMOTION_DIALOGUE_STRATEGIES: Dict[str, List[str]] = {
    "sadness": ["comfort", "encourage", "neutral"],
    "anger": ["calm", "neutral", "encourage"],
    "fear": ["reassure", "encourage", "neutral"],
    "joy": ["encourage", "celebrate", "neutral"],
    "none": ["neutral", "encourage", "comfort"],
}

STRATEGY_MESSAGES = {
    "comfort": "别灰心，失败是成功之母，我在这里陪着你。",
    "encourage": "你做得很好，继续加油！",
    "calm": "先深呼吸，冷静一下，我们慢慢来解决。",
    "reassure": "别怕，有我在，我们一起面对。",
    "celebrate": "太棒了！让我们一起庆祝这个时刻！",
    "neutral": "你好，今天想做什么？",
}


@dataclass
class Config:
    emotion_decay_rate: float = 0.1
    personality_stability: float = 0.9
    min_feedback_samples: int = 3


class PlayerPsychology:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._decay_rate = float(self.config.get("emotion_decay_rate", 0.1))
        self._stability = float(self.config.get("personality_stability", 0.9))
        self._min_feedback = int(self.config.get("min_feedback_samples", 3))
        self._state = {}
        self._history: List[str] = []
        self._emotions: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
        self._personality: Dict[str, float] = {t: 0.5 for t in BIG_FIVE}
        self._strategy_feedback: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # 基础接口
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """随时间推移情绪强度自然衰减。"""
        if dt <= 0:
            return
        decay = math.exp(-self._decay_rate * dt)
        for emotion in self._emotions:
            self._emotions[emotion] *= decay

    def reset(self):
        self._state = {}
        self._history = []
        self._emotions = {e: 0.0 for e in EMOTIONS}
        self._personality = {t: 0.5 for t in BIG_FIVE}
        self._strategy_feedback = {}

    def record_behavior(self, action: str):
        """记录一次玩家行为，驱动情绪、性格与预测模型更新。"""
        self._history.append(action)
        # 情绪：按行为特征加权累积
        for emotion, weight in BEHAVIOR_EMOTION_FEATURES.get(action, {}).items():
            self._emotions[emotion] = min(
                1.0, self._emotions[emotion] + weight * 0.5
            )
        # 性格：带稳定性的指数滑动平均，性格只能缓慢变化
        targets = BEHAVIOR_PERSONALITY_FEATURES.get(action)
        if targets:
            for trait in BIG_FIVE:
                target = targets.get(trait, 0.5)
                self._personality[trait] = (
                    self._stability * self._personality[trait]
                    + (1.0 - self._stability) * target
                )

    # ------------------------------------------------------------------
    # 1. 情绪模型：维度 + 强度
    # ------------------------------------------------------------------
    def emotion_model(self) -> Dict[str, float]:
        """返回各维度情绪的当前强度，取值 [0, 1]。"""
        return dict(self._emotions)

    def dominant_emotion(self) -> Tuple[str, float]:
        emotion, intensity = max(
            self._emotions.items(), key=lambda kv: kv[1]
        )
        if intensity <= 0.0:
            return "none", 0.0
        return emotion, intensity

    # ------------------------------------------------------------------
    # 2. 情绪识别：行为特征 + 置信度
    # ------------------------------------------------------------------
    def emotion_recognition(
        self, behaviors: Optional[List[str]] = None
    ) -> Dict:
        """从玩家行为序列推断情绪。

        返回主导情绪、各情绪概率分布、命中的行为特征以及置信度。
        置信度由证据一致性与证据数量共同决定。
        """
        if behaviors is None:
            behaviors = list(self._history)

        evidence = {e: 0.0 for e in EMOTIONS}
        matched_features: Dict[str, Dict[str, float]] = {}
        for action in behaviors:
            features = BEHAVIOR_EMOTION_FEATURES.get(action)
            if not features:
                continue
            matched_features[action] = dict(features)
            for emotion, weight in features.items():
                evidence[emotion] += weight

        total = sum(evidence.values())
        if total <= 0.0:
            return {
                "emotion": "none",
                "confidence": 0.0,
                "distribution": {e: 0.0 for e in EMOTIONS},
                "features": matched_features,
            }

        distribution = {e: evidence[e] / total for e in EMOTIONS}
        emotion = max(distribution, key=distribution.get)
        dominance = distribution[emotion]
        # 证据越多置信度越高，证据越一致置信度越高
        support = 1.0 - math.exp(-len(matched_features) / 3.0)
        confidence = dominance * support
        return {
            "emotion": emotion,
            "confidence": confidence,
            "distribution": distribution,
            "features": matched_features,
        }

    # ------------------------------------------------------------------
    # 3. 行为预测：马尔可夫模型 + 概率
    # ------------------------------------------------------------------
    def behavior_prediction(self) -> Dict:
        """基于历史行为的一阶马尔可夫模型预测下一步行为。

        返回最可能的下一行为、其概率以及完整的概率分布。
        """
        known_actions = sorted(BEHAVIOR_EMOTION_FEATURES.keys())
        if len(self._history) < 2:
            probability = 1.0 / len(known_actions)
            return {
                "action": None,
                "probability": probability,
                "distribution": {a: probability for a in known_actions},
            }

        last = self._history[-1]
        counts: Dict[str, int] = {}
        for prev, nxt in zip(self._history[:-1], self._history[1:]):
            if prev == last:
                counts[nxt] = counts.get(nxt, 0) + 1

        total = sum(counts.values())
        if total == 0:
            probability = 1.0 / len(known_actions)
            return {
                "action": None,
                "probability": probability,
                "distribution": {a: probability for a in known_actions},
            }

        distribution = {a: c / total for a, c in counts.items()}
        action = max(distribution, key=distribution.get)
        return {
            "action": action,
            "probability": distribution[action],
            "distribution": distribution,
        }

    # ------------------------------------------------------------------
    # 4. 性格模型：大五人格 + 稳定性
    # ------------------------------------------------------------------
    def personality_model(self) -> Dict:
        """返回大五人格各维度得分（[0, 1]）及稳定性系数。"""
        result = dict(self._personality)
        result["stability"] = self._stability
        return result

    # ------------------------------------------------------------------
    # 5. 自适应对话：策略 + 反馈
    # ------------------------------------------------------------------
    def adaptive_dialogue(self) -> Dict:
        """根据玩家当前情绪选择对话策略并生成对话内容。

        反馈记录会影响后续策略选择：效果差的策略会被降权。
        """
        emotion, intensity = self.dominant_emotion()
        candidates = list(EMOTION_DIALOGUE_STRATEGIES[emotion])

        def score(strategy: str) -> float:
            feedback = self._strategy_feedback.get(strategy, [])
            if len(feedback) < self._min_feedback:
                return 0.5  # 证据不足时的中性先验
            return sum(feedback) / len(feedback)

        # 在情绪对应的候选策略中，优先选择历史反馈最好的
        best = max(range(len(candidates)),
                   key=lambda i: (score(candidates[i]), -i))
        strategy = candidates[best]
        return {
            "strategy": strategy,
            "message": STRATEGY_MESSAGES[strategy],
            "emotion": emotion,
            "intensity": intensity,
        }

    def dialogue_feedback(self, strategy: str, effective: bool):
        """记录一次对话策略的效果反馈，用于后续策略调整。"""
        self._strategy_feedback.setdefault(strategy, []).append(
            1.0 if effective else 0.0
        )
