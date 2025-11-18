import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv


def create_app():
    # .env 로드
    load_dotenv()

    print("--- ★★★ app/__init__.py 의 create_app()이 호출되었습니다. ★★★ ---")

    app = Flask(__name__)

    # CORS 전체 허용 (localhost:3000 → 5000 호출 허용)
    CORS(
        app,
        resources={r"/*": {"origins": "*"}},
        supports_credentials=True,
    )

    # 🌟 기본 routes (기존 routes.py)
    from . import routes
    app.register_blueprint(routes.bp)

    # 🌟 효율적 프론티어 + 최적화 API
    from .api.optimize_api import optimize_api
    app.register_blueprint(optimize_api)

    # 🌟 🔥 AI 챗봇 API
    from .ai_chat import bp_ai
    app.register_blueprint(bp_ai)

    # 🌟 효율적 프론티어 전용 API (/api/frontier)
    from .api.frontier import bp as frontier_bp
    app.register_blueprint(frontier_bp)

    return app
