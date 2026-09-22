"""接口层：只做 HTTP（收请求、校验、序列化响应），业务全部下沉到 services/。"""
from flask import Blueprint, jsonify, render_template, request

from services.qa_service import handle_question

bp = Blueprint("web", __name__, template_folder="templates", static_folder="static")


@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@bp.route("/query", methods=["POST"])
def query():
    # silent=True：请求体不是合法 JSON 时返回 None，而不是抛 415
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question 不能为空"}), 400

    answer = handle_question(question)
    return jsonify({"answer": answer})


@bp.route("/health", methods=["GET"])
def health():
    """健康检查：脚本/监控探活用。"""
    return jsonify({"status": "ok"})
