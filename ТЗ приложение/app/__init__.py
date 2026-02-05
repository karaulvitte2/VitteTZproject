"""
Инициализация Flask-приложения.

Задачи этого модуля:
- создать объект Flask;
- настроить подключение к БД (SQLite);
- загрузить конфигурацию RAG-сервиса из JSON-файла;
- инициализировать RAGService (TF–IDF + корпус RAG + LLM-клиент);
- создать таблицы БД;
- зарегистрировать маршруты (blueprint из views.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from flask import Flask

from .rag_service import RAGService
from .models import db


def _load_rag_config(base_dir: Path) -> Dict[str, Any]:
    """
    Загружает конфигурацию RAG-сервиса из JSON-файла.

    Ожидается, что файл лежит в каталоге `flask_artifacts`
    рядом с артефактами TF–IDF и корпусом.
    """
    config_path = base_dir / "flask_artifacts" / "rag_flask_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Не найден конфигурационный файл RAG-сервиса: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def create_app() -> Flask:
    """
    Фабрика приложения Flask.

    Вызывается из main.py и при запуске через WSGI-сервер.
    """
    # __file__ → app/__init__.py → parent = app → parent = корень проекта
    base_dir = Path(__file__).resolve().parent.parent

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # ------------------------------------------------------------------
    # 1. Настройка БД (SQLite в корне проекта)
    # ------------------------------------------------------------------
    db_path = base_dir / "tz_generator.db"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    # ------------------------------------------------------------------
    # 2. Загрузка конфигурации и инициализация RAG-сервиса
    # ------------------------------------------------------------------
    rag_config = _load_rag_config(base_dir)
    rag_service = RAGService.from_config(base_dir=base_dir, config=rag_config)

    app.config["RAG_SERVICE"] = rag_service
    app.config["RAG_MODE_DEFAULT"] = rag_config["rag"]["mode_default"]

    # ------------------------------------------------------------------
    # 3. Создание таблиц БД
    # ------------------------------------------------------------------
    with app.app_context():
        # Импорт моделей нужен, чтобы SQLAlchemy "увидел" их перед create_all()
        from . import models  # noqa: F401
        db.create_all()

    # ------------------------------------------------------------------
    # 4. Регистрация blueprint с маршрутами
    # ------------------------------------------------------------------
    from .views import bp as main_bp

    app.register_blueprint(main_bp)

    return app
