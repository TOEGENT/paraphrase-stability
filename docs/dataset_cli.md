# Сборка и проверка датасета

## Подготовка и проверка

Если вопросы уже есть в CSV или TSV, импортируйте их отдельной командой:

```bash
python scripts/import_questions.py --input work/source.csv --question-column question --source pilot --output work/questions.jsonl
```

Для файла с колонкой идентификаторов добавьте `--id-column question_id`; разделитель задаётся через `--delimiter` (по умолчанию запятая для CSV и табуляция для TSV). Выход содержит `question_review: pending`: проверьте вопросы и пометьте нужные `approved` перед генерацией перефразов. Импорт проверяет пустые вопросы, точные дубликаты внутри файла и повторные ID; смысловые дубликаты по отношению к уже собранному набору остаются предметом ручной проверки.

После последнего ручного просмотра и команды `compute_stability.py --recompute --finalize` проверьте целостность и достаточность набора:

```bash
python scripts/check_dataset.py --input work/dataset_final.jsonl --min-questions 100 --min-variants-per-question 3 --min-total-variants 300 --report work/check_report.json
python scripts/split_dataset.py --input work/dataset_final.jsonl --output-dir work/splits --train 0.8 --val 0.1 --test 0.1 --seed 42
```

Минимумы, пропорции и seed задаёт человек для конкретного запуска; числа выше служат примером. Проверка возвращает код 1 при провале и сообщает причины в JSON. Разбиение принимает только утверждённые записи с согласованными `k`, `N`, `target` и сохраняет все перефразы вопроса вместе; создаёт `train.jsonl`, `val.jsonl`, `test.jsonl` и `split_manifest.json`, не перезаписывая непустой каталог. Для трёх непустых частей нужно минимум три вопроса.

Экспорт в формат `data/README.md` означал бы преобразование вложенного `dataset_final.jsonl` в пять отдельных файлов `questions.jsonl`, `answers.jsonl`, `rephrased_questions.jsonl`, `rephrased_answers.jsonl`, `equivalents.jsonl`. Текущие проверка и разбиение работают напрямую с итоговым JSONL, поэтому этот экспорт пока не нужен для описанного цикла; если следующие компоненты будут читать пять файлов, понадобится отдельный преобразователь. Аудит содержания вопросов, перефразов и решений оценщика остаётся ручным.

## Пять этапов генерации

Скрипты читают и пишут UTF-8 JSONL, один объект на строку. Каждый этап создаёт **новый** файл и отказывается перезаписывать существующий, чтобы правки человека не терялись. Запускать из корня репозитория после `pip install -e .`; для моделей используется `configs/model_tools.example.json` или файл, переданный через `--config`.

| Шаг | Команда | Что проверяет человек перед следующим шагом |
|---|---|---|
| 1. Исходные задачи | `python scripts/generate_questions.py --topic "арифметика" --count 5 --output work/questions.jsonl` | Текст, однозначность и смысловые дубликаты; меняет `question_review` с `pending` на `approved` или `rejected` |
| 2. Перефразы | `python scripts/generate_paraphrases.py --input work/questions_reviewed.jsonl --count 3 --output work/paraphrases.jsonl` | Сохранение фактов и требований; меняет `paraphrase_review` |
| 3. Оценка перефразов | `python scripts/evaluate_paraphrases.py --input work/paraphrases_reviewed.jsonl --output work/question_judgments.jsonl` | Решение оценщика в `question_equivalence`; при ошибке исправляет статус и меняет `equivalence_review` |
| 4. Ответы фиксированной SLM | `python scripts/generate_answers.py --input work/question_judgments_reviewed.jsonl --output work/answers.jsonl` | Скрипт пропускает отклонённые и неэквивалентные варианты, сохраняет исходный ответ и ответы на принятые перефразы |
| 5. Смысл ответов и `k/N` | `python scripts/compute_stability.py --input work/answers.jsonl --output work/dataset.jsonl` | Проверяет статусы ответов, `k`, `N`, текст и происхождение; отмечает итоговый `review_status` |

Файл с суффиксом `_reviewed` — это копия предыдущего файла после ручной проверки; скрипты сами не утверждают записи. Для сравнения новых задач с уже собранными добавьте к первому шагу `--existing data/processed/questions.jsonl`: точные совпадения и вероятные смысловые дубликаты помечаются в полях `duplicate_status` и `duplicate_of`, решение остаётся за человеком. На большой базе полный попарный смысловой поиск будет медленным; текущая реализация рассчитана на небольшой пилот.

Оценщик вызывается дважды для разных задач: на шаге 3 он проверяет смысл **вопросов**, а на шаге 5 — смысл пары **ответов**. На шаге 5 `N` равно числу утверждённых перефразов, для которых получен ответ SLM, а `k` — числу ответов со статусом `equivalent`; исходный вопрос в `N` не входит. Если есть `unclear` или `insufficient_context`, `target` остаётся `null` до ручного решения. После исправления статусов и установки `review_status` в `approved` или `rejected` пересчитайте метку и выпустите только утверждённые строки без вызова модели: `python scripts/compute_stability.py --recompute --finalize --input work/dataset_reviewed.jsonl --output work/dataset_final.jsonl`.

При правке JSONL сохраняйте `question_id` и `variant_id`: они связывают этапы. Статусы `pending` блокируют следующий этап, а `rejected` исключает запись. Скрипты проверяют уникальность ID на входе, но не доказывают качество работы генератора или оценщика; для исследовательского набора нужен ручной аудит и фиксация версий моделей и конфигурации.
