"""接口层：只做 HTTP（收请求、校验、序列化响应），业务全部下沉到 services/。"""
import json

from flask import Blueprint, Response, jsonify, render_template, request

from services.qa_service import handle_question, stream_question

bp = Blueprint("web", __name__, template_folder="templates", static_folder="static")


@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@bp.route("/query", methods=["POST"])
def query():
    """非流式问答：一次返回完整结果。"""
    # silent=True：请求体不是合法 JSON 时返回 None，而不是抛 415
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question 不能为空"}), 400

    result = handle_question(question)
    return jsonify({
        "answer": result["answer"],
        "elapsed": result["elapsed"],
        "used_agent": result["used_agent"],
    })


@bp.route("/query/stream", methods=["POST"])
def query_stream():
    """流式问答：以 SSE 逐步推送「检索结果」与「回答片段」。"""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question 不能为空"}), 400

    def event_stream():
        for event in stream_question(question):
            # SSE 格式：每行以 "data: " 开头，空行作为事件分隔
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # 关掉反向代理缓冲，保证逐块下发
        },
    )


@bp.route("/health", methods=["GET"])
def health():
    """健康检查：脚本/监控探活用。"""
    return jsonify({"status": "ok"})
