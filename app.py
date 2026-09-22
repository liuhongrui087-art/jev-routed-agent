"""入口：创建 app、注册蓝图、启动服务。只做这三件事。"""
from flask import Flask

import config
from web.routes import bp


def create_app():
    """应用工厂：便于测试与多环境复用。"""
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    print(f"启动服务 http://localhost:{config.PORT}/")
    create_app().run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
