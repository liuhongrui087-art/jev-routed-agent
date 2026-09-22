"""问句路由：判断问题该走哪条处理链。

两级路由：

  ① Jev（TypeSafe 决策模型）—— 首选
     专做决策的模型，输出带概率的类型化答案，不生成文本、无需解析。
     实测中文路由 6/6 正确、置信度 1.0、耗时约 1.1 秒。
     缺点：境外服务，中国大陆需代理；代理一断就不可用。

  ② 本地关键词规则 —— Jev 不可用时的兜底
     三个工具的字面线索本来就明确（"天气"、"计算"），规则足以覆盖，
     而且**永远稳定**。
     ⚠️ 关键：绝不能退回"带全部工具的 Agent 自主决策"——实测成功率只有约 1/3，
        那是整个项目最不稳的一环。规则判不准时，默认按概念题处理。

对外只有一个 route()，返回 (intent, confidence)。
"""
import requests

import config

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_MIN_CONFIDENCE = 0.6

# 本地规则命中时给的置信度（要 >= JEV_MIN_CONFIDENCE 才会被采用）
RULE_CONFIDENCE = 0.7

_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "这句话应该由哪种方式处理？",
        "criteria": {
            "knowledge": "机器学习的概念、术语、原理、方法相关的问题",
            "calculator": "需要做数学计算的问题",
            "weather": "查询某个地方的天气",
            "chat": "闲聊、问候，或与上述都无关的内容",
        },
    },
}

# ---------- 本地兜底规则的触发词 ----------

_WEATHER_WORDS = ("天气", "气温", "下雨", "下雪", "多少度", "冷不冷", "热不热")
_CALC_WORDS = ("计算", "算一下", "算算", "等于", "求和", "开方", "平方", "根号")
_CHAT_WORDS = ("你好", "您好", "hi", "hello", "谢谢", "再见", "在吗", "你是谁")


def _route_by_rules(question: str):
    """本地关键词规则。**永远返回结果**，判不出来就按概念题处理。

    顺序有讲究：先判"计算"再判"天气"——避免"北京今天多少度"被天气词先吃掉后
    又因含数字被误判；这里用"计算动词 + 数字"两个条件同时成立才算计算题。
    """
    q = question.strip().lower()

    # 计算题：必须有计算动词，且句子里出现数字
    if any(w in q for w in _CALC_WORDS) and any(c.isdigit() for c in q):
        return "calculator", RULE_CONFIDENCE

    if any(w in q for w in _WEATHER_WORDS):
        return "weather", RULE_CONFIDENCE

    if any(w in q for w in _CHAT_WORDS):
        return "chat", RULE_CONFIDENCE

    # 其余一律按概念题处理（本项目的知识库就是机器学习）
    return "knowledge", RULE_CONFIDENCE


def _route_by_jev(question: str):
    """Jev 决策。任何失败都返回 (None, 0.0)，绝不抛异常。"""
    key = getattr(config, "TYPESAFE_API_KEY", "")
    if not key:
        print("[router] 未配置 TYPESAFE_API_KEY")
        return None, 0.0

    try:
        resp = requests.post(
            JEV_URL,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
            json={"model": JEV_MODEL, "state": question, "questions": _QUESTIONS},
            timeout=8,
        )
        resp.raise_for_status()
        ans = resp.json()["answers"]["intent"]
        return ans.get("choice"), float(ans.get("confidence", 0.0))
    except Exception as e:
        print("[router] Jev 调用失败: " + type(e).__name__ + ": " + str(e))
        return None, 0.0


def route(question: str):
    """判断问题类型。返回 (intent, confidence)。

    先试 Jev；Jev 不可用或置信度不足时，用本地规则兜底。
    两个来源都会打印日志，方便判断当前走的是哪一级。
    """
    intent, confidence = _route_by_jev(question)
    if intent and confidence >= JEV_MIN_CONFIDENCE:
        print("[router] 来源 = Jev")
        return intent, confidence

    intent, confidence = _route_by_rules(question)
    print("[router] 来源 = 本地规则")
    return intent, confidence
