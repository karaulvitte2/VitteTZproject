"""
Маршруты Flask-приложения (уровень представления).

Здесь:
- главная страница с формой генерации раздела ТЗ;
- обработка генерации (вызов RAGService + запись в БД);
- просмотр журнала и конструктор ТЗ;
- формирование и скачивание DOCX-документов;
- страница «О сервисе».
"""

from __future__ import annotations

from typing import List, Dict, Any

from flask import (
    Blueprint,
    current_app,
    render_template,
    request,
    send_file,
)

from .models import db, GenerationLog, Document, DocumentSection
from .docx_utils import build_docx_from_logs

bp = Blueprint("main", __name__)

# ---------------------------- Тестовые проекты -----------------------------


TEST_PROJECTS: List[Dict[str, Any]] = [
    {
        "id": "hr_muiv",
        "name": "Система учета сотрудников",
        "domain": "учет кадров в вузе",
        "description": (
            "Разработка информационной системы учета сотрудников "
            "частного образовательного учреждения высшего образования "
            "«Московский университет имени С. Ю. Витте». "
            "Система должна хранить сведения о штатном и совместительском "
            "персонале, поддерживать поиск по должности и подразделению, "
            "формировать отчеты для кадровой службы и руководства."
        ),
    },
    {
        "id": "vkr_support",
        "name": "Система поддержки ВКР",
        "domain": "образовательные процессы в вузе",
        "description": (
            "Создание информационной системы поддержки подготовки и хранения "
            "выпускных квалификационных работ студентов. Система позволяет "
            "регистрировать темы ВКР, закреплять научных руководителей, "
            "загружать промежуточные и итоговые версии работ, а также "
            "обеспечивать доступ к архиву ВКР с учетом ролей пользователей."
        ),
    },
    {
        "id": "edo_department",
        "name": "Подсистема электронного документооборота кафедры",
        "domain": "электронный документооборот",
        "description": (
            "Разработка подсистемы электронного документооборота (ЭДО) "
            "для кафедры в составе корпоративной информационной системы вуза. "
            "Подсистема обеспечивает регистрацию, согласование и хранение "
            "служебных записок, приказов, заявлений и других документов, "
            "поддерживает контроль сроков исполнения и разграничение прав доступа."
        ),
    },
]

# Доступные разделы ТЗ
TZ_SECTIONS: List[str] = [
    "Основания для разработки",
    "Назначение системы",
    "Требования к системе",
]

# Режимы работы RAG/LLM
TZ_MODES: List[str] = [
    "baseline",
    "rag_gost",
    "rag_full",
]


def _get_rag_service():
    """Удобный короткий доступ к сервису из config приложения."""
    rag_service = current_app.config.get("RAG_SERVICE")
    if rag_service is None:
        raise RuntimeError("RAG_SERVICE не инициализирован. Проверьте create_app().")
    return rag_service


# ----------------------------------------------------------------------------
# Главная страница: форма генерации раздела ТЗ
# ----------------------------------------------------------------------------


@bp.route("/", methods=["GET"])
def index():
    """
    Главная страница с формой генерации раздела технического задания.
    """
    default_project = TEST_PROJECTS[0]

    context = {
        "projects": TEST_PROJECTS,
        "selected_project_id": default_project["id"],
        "project_name": default_project["name"],
        "project_domain": default_project["domain"],
        "project_description": default_project["description"],
        "tz_sections": TZ_SECTIONS,
        "selected_section": "Назначение системы",
        "tz_modes": TZ_MODES,
        "selected_mode": current_app.config.get("RAG_MODE_DEFAULT", "rag_gost"),
        "generated_text": None,
        "used_chunks": None,
    }

    return render_template("index.html", **context)


@bp.route("/generate", methods=["POST"])
def generate():
    """
    Обработка формы и генерация раздела ТЗ.

    1. Считывает поля формы (проект, описание, раздел, режим).
    2. Вызывает RAG-сервис для генерации текста.
    3. Сохраняет результат в базу данных (таблица GenerationLog).
    4. Отображает результат на главной странице.
    """
    rag_service = _get_rag_service()

    # 1. Данные формы
    project_id = request.form.get("project_id") or ""
    project_name = request.form.get("project_name") or ""
    project_domain = request.form.get("project_domain") or ""
    project_description = request.form.get("project_description") or ""
    section_name = request.form.get("section_name") or "Назначение системы"
    mode = request.form.get("mode") or current_app.config.get("RAG_MODE_DEFAULT", "rag_gost")

    # Если выбран один из тестовых проектов и описание пустое —
    # подставляем заранее заготовленное описание.
    selected_project = next(
        (p for p in TEST_PROJECTS if p["id"] == project_id),
        None,
    )

    if selected_project is not None and not project_description.strip():
        project_name = selected_project["name"]
        project_domain = selected_project["domain"]
        project_description = selected_project["description"]

    # 2. Генерация раздела через RAG-сервис
    result = rag_service.generate_section(
        project_name=project_name,
        project_domain=project_domain,
        project_description=project_description,
        section_name=section_name,
        mode=mode,
    )

    generated_text = result.get("text", "")
    used_chunks = result.get("used_chunks", [])

    # 3. Запись результата в базу данных
    try:
        log_entry = GenerationLog(
            project_name=project_name,
            project_domain=project_domain,
            section_name=section_name,
            mode=mode,
            generated_text=generated_text,
        )
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e:
        # В учебном проекте достаточно вывести сообщение в консоль;
        # при желании можно добавить логирование в файл.
        print(f"[DB ERROR] Не удалось сохранить запись GenerationLog: {e}")

    # 4. Формируем контекст для шаблона
    context = {
        "projects": TEST_PROJECTS,
        "selected_project_id": project_id or (selected_project["id"] if selected_project else ""),
        "project_name": project_name,
        "project_domain": project_domain,
        "project_description": project_description,
        "tz_sections": TZ_SECTIONS,
        "selected_section": section_name,
        "tz_modes": TZ_MODES,
        "selected_mode": mode,
        "generated_text": generated_text,
        "used_chunks": used_chunks,
    }

    return render_template("index.html", **context)


# ----------------------------------------------------------------------------
# Просмотр журнала и конструктор ТЗ
# ----------------------------------------------------------------------------


@bp.route("/history", methods=["GET"])
def history():
    """
    Страница просмотра журнала генерации и конструктор документа ТЗ.

    Передаёт в шаблон:
    - history_entries: список записей GenerationLog
    - documents: список ранее сформированных документов ТЗ (Document)
    """
    # Берём, например, последние 100 записей журнала
    history_entries = (
        GenerationLog.query
        .order_by(GenerationLog.created_at.desc())
        .limit(100)
        .all()
    )

    # И последние сформированные документы ТЗ
    documents = (
        Document.query
        .order_by(Document.created_at.desc())
        .limit(50)
        .all()
    )

    context = {
        "history_entries": history_entries,
        "documents": documents,
        # Поля формы для сборки документа (по умолчанию — пустые)
        "doc_title": "",
        "doc_project_name": "",
        "doc_project_domain": "",
        "doc_comment": "",
    }

    return render_template("history.html", **context)


@bp.route("/history/build", methods=["POST"])
def build_document():
    """
    Обработка формы конструктора ТЗ.

    1. Получает список log_ids (отмеченные записи журнала).
    2. Загружает соответствующие записи GenerationLog.
    3. Создаёт запись Document и DocumentSection в БД.
    4. Собирает DOCX-документ и отдаёт его пользователю.
    """
    # 1. Список отмеченных записей журнала
    log_id_values = request.form.getlist("log_ids")

    if not log_id_values:
        # Если ничего не выбрано — просто возвращаем страницу history
        # (без сообщения об ошибке, чтобы не усложнять интерфейс).
        return history()

    try:
        log_ids = [int(x) for x in log_id_values]
    except ValueError:
        return history()

    # 2. Загружаем записи журнала из БД
    sections = (
        GenerationLog.query
        .filter(GenerationLog.id.in_(log_ids))
        .all()
    )

    if not sections:
        return history()

    # 3. Читаем параметры итогового документа из формы
    doc_title = request.form.get("doc_title") or "Техническое задание"
    doc_project_name = request.form.get("doc_project_name") or sections[0].project_name
    doc_project_domain = request.form.get("doc_project_domain") or sections[0].project_domain
    doc_comment = request.form.get("doc_comment") or ""

    # 4. Сохраняем метаданные документа и связи "документ ↔ разделы"
    document = Document(
        title=doc_title,
        project_name=doc_project_name,
        project_domain=doc_project_domain,
        comment=doc_comment,
    )
    db.session.add(document)
    db.session.flush()  # чтобы получить document.id до commit

    # Порядок разделов внутри документа (по порядку выборки);
    # фактический порядок при формировании DOCX будет уточнён в docx_utils.
    order_index = 1
    for log_entry in sections:
        link = DocumentSection(
            document_id=document.id,
            log_id=log_entry.id,
            section_name=log_entry.section_name,
            order_index=order_index,
        )
        order_index += 1
        db.session.add(link)

    db.session.commit()

    # 5. Собираем DOCX из выбранных разделов
    buffer = build_docx_from_logs(
        sections=sections,
        doc_title=document.title,
        project_name=document.project_name,
        project_domain=document.project_domain,
        comment=document.comment,
    )

    # Формируем безопасное имя файла (простейший вариант)
    safe_title = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in document.title)
    filename = (safe_title or "TZ_document").strip().replace(" ", "_") + ".docx"

    return send_file(
        buffer,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        as_attachment=True,
        download_name=filename,
    )


@bp.route("/documents/<int:document_id>/download", methods=["GET"])
def download_document(document_id: int):
    """
    Повторная выгрузка ранее сформированного документа ТЗ.

    1. Находит Document по идентификатору.
    2. Находит связанные DocumentSection и соответствующие записи GenerationLog.
    3. Собирает DOCX и отдаёт файл пользователю.
    """
    document = Document.query.get(document_id)
    if document is None:
        # В учебном проекте можно просто вернуть главную страницу или 404,
        # здесь выберем возврат на history.
        return history()

    # Находим связанные секции документа
    section_links = (
        DocumentSection.query
        .filter(DocumentSection.document_id == document.id)
        .order_by(DocumentSection.order_index.asc())
        .all()
    )

    if not section_links:
        return history()

    log_ids = [link.log_id for link in section_links]

    sections = (
        GenerationLog.query
        .filter(GenerationLog.id.in_(log_ids))
        .all()
    )

    if not sections:
        return history()

    buffer = build_docx_from_logs(
        sections=sections,
        doc_title=document.title,
        project_name=document.project_name,
        project_domain=document.project_domain,
        comment=document.comment,
    )

    safe_title = "".join(ch if ch.isalnum() or ch in (" ", "_", "-") else "_" for ch in document.title)
    filename = (safe_title or f"TZ_document_{document.id}").strip().replace(" ", "_") + ".docx"

    return send_file(
        buffer,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        as_attachment=True,
        download_name=filename,
    )


# ----------------------------------------------------------------------------
# Страница «О сервисе»
# ----------------------------------------------------------------------------


@bp.route("/about", methods=["GET"])
def about():
    """
    Страница «О сервисе».

    Здесь выводится краткая справка о назначении системы, используемых
    технологиях и связи с преддипломной практикой.
    """
    return render_template("about.html")
