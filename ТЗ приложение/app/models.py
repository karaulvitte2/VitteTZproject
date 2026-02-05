"""
Модели базы данных для Flask-приложения.

Используются три сущности:
- GenerationLog      — журнал сгенерированных разделов ТЗ;
- Document           — метаданные собранного документа ТЗ в целом;
- DocumentSection    — связь "документ ТЗ ↔ записи журнала" с указанием порядка.
"""

from __future__ import annotations

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

# Единственный экземпляр SQLAlchemy, инициализируется в create_app()
db = SQLAlchemy()


class GenerationLog(db.Model):
    """
    Журнал генерации разделов ТЗ.

    Каждая запись соответствует одному вызову генерации:
    - конкретный проект;
    - название раздела ТЗ;
    - режим работы RAG/LLM;
    - сгенерированный текст.
    """
    __tablename__ = "generation_log"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    project_name = db.Column(db.String(255), nullable=False)
    project_domain = db.Column(db.String(255), nullable=True)

    section_name = db.Column(db.String(255), nullable=False)
    mode = db.Column(db.String(50), nullable=False)

    generated_text = db.Column(db.Text, nullable=False)

    # Связь с DocumentSection (из каких записей журнала собран документ ТЗ)
    document_sections = db.relationship(
        "DocumentSection",
        back_populates="log_entry",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<GenerationLog id={self.id} "
            f"project={self.project_name!r} "
            f"section={self.section_name!r} "
            f"mode={self.mode!r}>"
        )


class Document(db.Model):
    """
    Документ ТЗ в целом.

    Хранит метаданные "собранного" технического задания:
    - заголовок документа;
    - проект и предметную область;
    - комментарий (заметка: версия, назначение, статус).
    """
    __tablename__ = "document"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    title = db.Column(db.String(255), nullable=False)
    project_name = db.Column(db.String(255), nullable=False)
    project_domain = db.Column(db.String(255), nullable=True)

    comment = db.Column(db.Text, nullable=True)

    # Связанные разделы документа (через таблицу document_section)
    sections = db.relationship(
        "DocumentSection",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentSection.order_index",
    )

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id} "
            f"title={self.title!r} "
            f"project={self.project_name!r}>"
        )


class DocumentSection(db.Model):
    """
    Связь "документ ТЗ ↔ запись журнала GenerationLog".

    Позволяет:
    - указать, какие разделы входят в документ;
    - задать порядок разделов внутри документа.
    """
    __tablename__ = "document_section"

    id = db.Column(db.Integer, primary_key=True)

    document_id = db.Column(
        db.Integer,
        db.ForeignKey("document.id", ondelete="CASCADE"),
        nullable=False,
    )
    log_id = db.Column(
        db.Integer,
        db.ForeignKey("generation_log.id", ondelete="CASCADE"),
        nullable=False,
    )

    section_name = db.Column(db.String(255), nullable=False)
    order_index = db.Column(db.Integer, nullable=False, default=1)

    # Обратные связи
    document = db.relationship(
        "Document",
        back_populates="sections",
    )
    log_entry = db.relationship(
        "GenerationLog",
        back_populates="document_sections",
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentSection id={self.id} "
            f"document_id={self.document_id} "
            f"log_id={self.log_id} "
            f"section={self.section_name!r} "
            f"order_index={self.order_index}>"
        )
