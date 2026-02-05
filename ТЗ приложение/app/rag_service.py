"""
Сервисный слой для работы с RAG и LLM.

Задачи RAGService:
- загрузить артефакты (корпус чанков, TF–IDF-векторизатор, TF–IDF-матрицу);
- по конфигу понимать режимы работы (baseline / rag_gost / rag_full);
- по запросу генерировать раздел ТЗ, используя Proxy API ChatGPT.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import load
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RAGConfig:
    """Упрощённое представление конфигурации RAG-сервиса."""
    model_name: str
    proxyapi_base_url: str
    rag_corpus_path: Path
    tfidf_vectorizer_path: Path
    tfidf_matrix_path: Path
    mode_default: str
    top_k_chunks_default: int
    modes: Dict[str, Dict[str, Any]]


class RAGService:
    """
    Класс, инкапсулирующий логику:
    - поиска релевантных чанков по TF–IDF;
    - формирования промптов;
    - вызова LLM через Proxy API;
    - получения текста раздела ТЗ.
    """

    def __init__(
        self,
        config: RAGConfig,
        corpus_chunks: List[Dict[str, Any]],
        tfidf_vectorizer,
        tfidf_matrix,
        llm_client: OpenAI,
    ) -> None:
        self.config = config
        self.corpus_chunks = corpus_chunks
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.llm_client = llm_client

    # -------------------------------------------------------------------------
    # Инициализация из конфигурационного JSON
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, base_dir: Path, config: Dict[str, Any]) -> "RAGService":
        """
        Создаёт экземпляр сервиса на основе словаря конфигурации,
        загруженного из rag_flask_config.json.
        """
        llm_conf = config.get("llm", {})
        rag_conf = config.get("rag", {})

        model_name = llm_conf.get("model_name", "gpt-4o")
        proxyapi_base_url = llm_conf.get("proxyapi_base_url", "https://openai.api.proxyapi.ru/v1")

        # Пути к артефактам задаём относительно корня проекта
        rag_corpus_path = base_dir / rag_conf.get("rag_corpus_path", "rag_corpus/rag_corpus_chunks.jsonl")
        tfidf_vectorizer_path = base_dir / rag_conf.get("tfidf_vectorizer_path", "flask_artifacts/tfidf_vectorizer.joblib")
        tfidf_matrix_path = base_dir / rag_conf.get("tfidf_matrix_path", "flask_artifacts/tfidf_matrix.joblib")

        mode_default = rag_conf.get("mode_default", "rag_gost")
        top_k_chunks_default = int(rag_conf.get("top_k_chunks_default", 8))
        modes = rag_conf.get("modes", {})

        ragcfg = RAGConfig(
            model_name=model_name,
            proxyapi_base_url=proxyapi_base_url,
            rag_corpus_path=rag_corpus_path,
            tfidf_vectorizer_path=tfidf_vectorizer_path,
            tfidf_matrix_path=tfidf_matrix_path,
            mode_default=mode_default,
            top_k_chunks_default=top_k_chunks_default,
            modes=modes,
        )

        # 1. Загружаем корпус чанков (JSONL)
        corpus_chunks = cls._load_corpus_chunks(ragcfg.rag_corpus_path)

        # 2. Загружаем TF–IDF-векторизатор и матрицу
        tfidf_vectorizer = load(ragcfg.tfidf_vectorizer_path)
        tfidf_matrix = load(ragcfg.tfidf_matrix_path)

        # 3. Инициализируем LLM-клиент Proxy API
        api_key = (
            os.getenv("PROXYAPI_KEY")
            or os.getenv("LITELLM_API_KEY")
            or "sk-Y2VSk9ZKuCJbQD9xO3jp0jVxlJsGynOz"  # тестовый ключ из задания
        )
        if not api_key:
            raise RuntimeError(
                "Не найден ключ PROXYAPI_KEY (или LITELLM_API_KEY) в переменных окружения. "
                "Установите его перед запуском приложения."
            )

        llm_client = OpenAI(
            api_key=api_key,
            base_url=ragcfg.proxyapi_base_url,
        )

        return cls(
            config=ragcfg,
            corpus_chunks=corpus_chunks,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            llm_client=llm_client,
        )

    # -------------------------------------------------------------------------
    # Загрузка корпуса
    # -------------------------------------------------------------------------
    @staticmethod
    def _load_corpus_chunks(path: Path) -> List[Dict[str, Any]]:
        """
        Загружает список чанков из JSONL-файла.
        Важно сохранять порядок строк, т.к. он должен совпадать
        с порядком в TF–IDF-матрице.
        """
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл корпуса RAG: {path}")

        chunks: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(obj)

        return chunks

    # -------------------------------------------------------------------------
    # Публичный метод: генерация раздела ТЗ для веб-интерфейса
    # -------------------------------------------------------------------------
    def generate_section(
        self,
        project_name: str,
        project_domain: str,
        project_description: str,
        section_name: str,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Генерирует текст раздела ТЗ для веб-интерфейса.

        Параметры:
        - project_name: название проекта;
        - project_domain: краткое описание предметной области;
        - project_description: текстовое описание системы;
        - section_name: название раздела ТЗ;
        - mode: режим (baseline / rag_gost / rag_full).

        Возвращает словарь:
        {
          "text": <сгенерированный текст>,
          "used_chunks": [список chunk_id]
        }
        """
        if not mode:
            mode = self.config.mode_default

        mode_cfg = self.config.modes.get(mode)
        if mode_cfg is None:
            raise ValueError(f"Неизвестный режим генерации: {mode!r}")

        use_rag = bool(mode_cfg.get("use_rag", False))
        allowed_source_types = mode_cfg.get("allowed_source_types")

        # 1. Формируем текстовый запрос для ретривера, если нужен RAG
        retrieved_chunks: List[Dict[str, Any]] = []
        if use_rag:
            query_text = self._build_retrieval_query(
                project_name=project_name,
                project_domain=project_domain,
                project_description=project_description,
                section_name=section_name,
            )
            retrieved_chunks = self._retrieve_chunks(
                query_text=query_text,
                top_k=self.config.top_k_chunks_default,
                allowed_source_types=allowed_source_types,
            )

        # 2. Формируем промпты
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            project_name=project_name,
            project_domain=project_domain,
            project_description=project_description,
            section_name=section_name,
            retrieved_chunks=retrieved_chunks,
            use_rag=use_rag,
        )

        # 3. Вызов LLM
        answer_text = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return {
            "text": answer_text,
            "used_chunks": [ch["chunk_id"] for ch in retrieved_chunks],
        }

    # -------------------------------------------------------------------------
    # Внутренняя логика: ретривер
    # -------------------------------------------------------------------------
    def _build_retrieval_query(
        self,
        project_name: str,
        project_domain: str,
        project_description: str,
        section_name: str,
    ) -> str:
        """
        Формирует текстовый запрос для TF–IDF ретривера.
        """
        return (
            f"Проект: {project_name}\n"
            f"Предметная область: {project_domain}\n\n"
            f"Описание проекта:\n{project_description}\n\n"
            f"Нужно сформировать раздел ТЗ: {section_name}.\n"
            "Техническое задание на разработку информационной системы в вузе."
        )

    def _retrieve_chunks(
        self,
        query_text: str,
        top_k: int = 8,
        allowed_source_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ищет наиболее релевантные чанки с помощью TF–IDF и косинусного сходства.
        Если задан allowed_source_types — фильтрует источники по типу.
        """
        if not query_text.strip():
            return []

        # Вектор запроса
        query_vec = self.tfidf_vectorizer.transform([query_text])

        # Сходство запрос↔чанки
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]  # shape: (n_chunks,)

        # Фильтр по типу источников
        if allowed_source_types is not None:
            allowed_set = set(allowed_source_types)
            for i, ch in enumerate(self.corpus_chunks):
                if ch.get("source_type") not in allowed_set:
                    similarities[i] = -1.0

        # Индексы top_k по убыванию
        top_k = max(1, min(top_k, len(self.corpus_chunks)))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(similarities[idx])
            ch = self.corpus_chunks[idx]
            results.append(
                {
                    "chunk_id": ch.get("chunk_id", f"chunk_{idx}"),
                    "doc_id": ch.get("doc_id", ""),
                    "source_type": ch.get("source_type", ""),
                    "title": ch.get("title", ""),
                    "url": ch.get("url", ""),
                    "chunk_index": ch.get("chunk_index", idx),
                    "score": score,
                    "text": ch.get("text", ""),
                }
            )

        return results

    # -------------------------------------------------------------------------
    # Внутренняя логика: построение промптов
    # -------------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        """
        Системный промпт: роль модели — эксперт по ГОСТ 19.201-78 и ТЗ.
        """
        return (
            "Ты — эксперт по стандартизации и проектированию информационных систем.\n"
            "Твоя задача — формировать разделы технического задания (ТЗ) в соответствии "
            "с ГОСТ 19.201-78 и практикой разработки автоматизированных систем в вузах.\n\n"
            "Требования к ответу:\n"
            "1. Пиши по-русски, академичным, но понятным языком.\n"
            "2. Соблюдай структуру и терминологию ГОСТ 19.201-78.\n"
            "3. Учитывай, что объект автоматизации — информационная система в университете.\n"
            "4. Не выдумывай факты о конкретном университете, если они не указаны во входных данных.\n"
            "5. Формируй текст так, чтобы его можно было сразу вставить в раздел ТЗ.\n"
        )

    def _build_user_prompt(
        self,
        project_name: str,
        project_domain: str,
        project_description: str,
        section_name: str,
        retrieved_chunks: List[Dict[str, Any]],
        use_rag: bool,
    ) -> str:
        """
        Пользовательский промпт: описание проекта + название раздела ТЗ
        + (опционально) контекст из RAG.
        """
        base_prompt = (
            f"Проект: {project_name}\n"
            f"Предметная область: {project_domain}\n\n"
            f"Краткое описание проекта:\n{project_description}\n\n"
            f"Необходимо сформировать раздел технического задания (ТЗ): «{section_name}».\n\n"
            "Опиши данный раздел так, как это принято в ГОСТ 19.201-78, с учётом того, "
            "что система создаётся для вуза. Следи за связностью текста и логикой изложения, "
            "избегай излишней воды и общих фраз."
        )

        if use_rag and retrieved_chunks:
            context_parts: List[str] = []
            for i, ch in enumerate(retrieved_chunks, start=1):
                chunk_text = ch["text"]
                if len(chunk_text) > 800:
                    chunk_text = chunk_text[:800] + "…"

                context_parts.append(
                    f"[Фрагмент {i} | источник: {ch['source_type']} | документ: {ch['title']}]\n"
                    f"{chunk_text}"
                )

            context_block = "\n\n".join(context_parts)

            user_prompt = (
                base_prompt
                + "\n\n--- Контекст (фрагменты из ГОСТ и связанных документов) ---\n"
                + context_block
                + "\n\nИспользуя приведённый контекст, сформируй связный и аккуратный текст "
                  "раздела ТЗ. При необходимости переформулируй фрагменты, не копируй их дословно."
            )
        else:
            user_prompt = (
                base_prompt
                + "\n\nКонтекст по ГОСТ и методическим материалам явно не подставляется. "
                  "Ориентируйся на общие требования к структуре ТЗ в соответствии с ГОСТ 19.201-78."
            )

        return user_prompt

    # -------------------------------------------------------------------------
    # Внутренняя логика: вызов LLM
    # -------------------------------------------------------------------------
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Вызывает модель Proxy API ChatGPT и возвращает текст ответа.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.2,
            )
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            # В учебном проекте достаточно текстового сообщения,
            # в реальном сервисе лучше логировать ошибку.
            return (
                "Ошибка при обращении к модели LLM. "
                f"Текст ошибки: {e}"
            )
