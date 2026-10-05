"""
游戏内玩家心理模型与情感计算系统 - 验收测试
运行方式：python -m pytest tests/test_player_psychology.py -q
共 15 个测试用例
"""
try:
    import pytest
except ImportError:
    pytest = None
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.player_psychology import PlayerPsychology


class TestPlayerPsychology:
    def setup_method(self):
        self.system = PlayerPsychology()

    # ---------------- 情绪模型：维度 + 强度 ----------------

    def test_case_01(self):
        # 情绪模型包含快乐/愤怒/悲伤/恐惧四个维度
        model = self.system.emotion_model()
        for dim in ("joy", "anger", "sadness", "fear"):
            assert dim in model

    def test_case_02(self):
        # 情绪有强度，且强度取值在 [0, 1]
        for action in ("attack", "die", "complete_quest", "flee"):
            self.system.record_behavior(action)
        model = self.system.emotion_model()
        assert all(0.0 <= v <= 1.0 for v in model.values())
        assert any(v > 0.0 for v in model.values())

    def test_case_03(self):
        # 情绪强度随行为变化，并随时间衰减
        self.system.record_behavior("attack")
        anger_before = self.system.emotion_model()["anger"]
        assert anger_before > 0.0
        self.system.update(10.0)
        assert self.system.emotion_model()["anger"] < anger_before

    # ---------------- 情绪识别：特征 + 置信度 ----------------

    def test_case_04(self):
        # 能从玩家行为特征推断情绪
        result = self.system.emotion_recognition(["attack", "destroy"])
        assert result["emotion"] == "anger"
        assert "attack" in result["features"]

    def test_case_05(self):
        # 识别结果带置信度，且取值在 [0, 1]
        result = self.system.emotion_recognition(["flee", "hide"])
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["confidence"] > 0.0

    def test_case_06(self):
        # 证据越一致、越多，置信度越高
        weak = self.system.emotion_recognition(["flee"])
        strong = self.system.emotion_recognition(
            ["flee", "hide", "avoid", "die"]
        )
        assert strong["confidence"] > weak["confidence"]
        assert strong["emotion"] in ("fear", "sadness")

    # ---------------- 行为预测：模型 + 概率 ----------------

    def test_case_07(self):
        # 能根据历史行为预测下一步行为
        for _ in range(5):
            self.system.record_behavior("explore")
            self.system.record_behavior("complete_quest")
        self.system.record_behavior("explore")
        prediction = self.system.behavior_prediction()
        assert prediction["action"] == "complete_quest"

    def test_case_08(self):
        # 预测结果带概率，且概率取值在 [0, 1]
        for _ in range(3):
            self.system.record_behavior("chat")
            self.system.record_behavior("help")
        self.system.record_behavior("chat")
        prediction = self.system.behavior_prediction()
        assert 0.0 <= prediction["probability"] <= 1.0
        assert prediction["probability"] > 0.0

    def test_case_09(self):
        # 概率分布归一化，且重复模式给出高置信预测
        for _ in range(4):
            self.system.record_behavior("attack")
            self.system.record_behavior("flee")
        self.system.record_behavior("attack")
        prediction = self.system.behavior_prediction()
        total = sum(prediction["distribution"].values())
        assert abs(total - 1.0) < 1e-9
        assert prediction["probability"] >= 0.5

    # ---------------- 性格模型：大五人格 + 稳定性 ----------------

    def test_case_10(self):
        # 性格模型包含大五人格五个维度
        model = self.system.personality_model()
        for trait in ("openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"):
            assert trait in model

    def test_case_11(self):
        # 不同行为历史的玩家性格不同，且得分在 [0, 1]
        for _ in range(5):
            self.system.record_behavior("explore")
        model = self.system.personality_model()
        assert all(0.0 <= model[t] <= 1.0 for t in model if t != "stability")
        assert model["openness"] > 0.5

    def test_case_12(self):
        # 性格具有稳定性：单次行为不会让性格剧烈变化
        before = self.system.personality_model()["extraversion"]
        self.system.record_behavior("chat")
        after = self.system.personality_model()["extraversion"]
        assert abs(after - before) < 0.2
        assert self.system.personality_model()["stability"] > 0.0

    # ---------------- 自适应对话：策略 + 反馈 ----------------

    def test_case_13(self):
        # NPC 根据玩家情绪选择对话策略：悲伤时安慰
        self.system.record_behavior("die")
        dialogue = self.system.adaptive_dialogue()
        assert dialogue["strategy"] == "comfort"
        assert dialogue["message"]

    def test_case_14(self):
        # 不同情绪触发不同对话策略：愤怒时安抚、开心时鼓励
        self.system.record_behavior("attack")
        assert self.system.adaptive_dialogue()["strategy"] == "calm"
        self.system.reset()
        self.system.record_behavior("complete_quest")
        assert self.system.adaptive_dialogue()["strategy"] == "encourage"

    def test_case_15(self):
        # 对话有反馈机制：持续负反馈会让 NPC 更换策略
        self.system.record_behavior("die")
        assert self.system.adaptive_dialogue()["strategy"] == "comfort"
        for _ in range(5):
            self.system.dialogue_feedback("comfort", effective=False)
            self.system.dialogue_feedback("encourage", effective=True)
        dialogue = self.system.adaptive_dialogue()
        assert dialogue["strategy"] == "encourage"


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        _t = TestPlayerPsychology
        _p = _f = 0
        for _n in dir(_t):
            if _n.startswith("test_case"):
                _i = _t()
                _i.setup_method()
                try:
                    getattr(_i, _n)()
                    _p += 1
                except Exception as _e:
                    _f += 1
                    print(_n, "FAILED:", _e)
        print(f"{_p} passed, {_f} failed")
