# Контракты скриптов

Здесь описаны реализованные файловые интерфейсы `scripts/*.py`. Команды запуска приведены в [dataset_cli.md](dataset_cli.md) и [mlp_pipeline.md](mlp_pipeline.md).

## Общие правила

- JSONL содержит по одному объекту UTF-8 на строку; пустой входной JSONL отклоняется.
- `question_id` связывает вопрос с вариантами; `variant_id` идентифицирует вариант.
- Выходные файлы и непустые выходные каталоги не перезаписываются. `generate_paraphrases.py` дополнительно дописывает журнал `--log-file`.
- `pending` на очередной ручной проверке останавливает следующий этап; `rejected` исключает запись; `approved` пропускает её дальше.
- Ниже указаны обязательные для работы поля. Необязательные `source`, `topic`, `language` и сведения о моделях и преобразованиях передаются дальше, когда присутствуют.

## Создание датасета

| Скрипт | Входные данные | Выходные данные |
|---|---|---|
| `import_questions.py` | `--input`: CSV/TSV с колонкой `question` (имя задаёт `--question-column`); необязательные колонка ID (`--id-column`), `topic`, `language`; `--source`. | `--output`: JSONL с `question_id`, `question`, `source`, `duplicate_status`, `duplicate_of`, `question_review: pending`; переносит `topic` и `language`, если они есть. |
| `generate_questions.py` | `--topic`, `--count`, конфигурация генератора; необязательный `--existing`: JSONL с полем `question` для проверки повторов. | `--output`: JSONL с `question_id`, `question`, `source`, `topic`, `generator_model`, `duplicate_status`, `duplicate_of`, `question_review: pending`. |
| `generate_paraphrases.py` | `--input`: JSONL с уникальными `question_id`, полями `question`, `question_review: approved/rejected`; `--count` новых вариантов на утверждённый вопрос, режим и конфигурация генератора; необязательный `--existing`: варианты с `variant_id`, `question_id`, `question_original`, `question_transformed`. | `--output`: только новые варианты в JSONL с `question_id`, `variant_id`, `question_original`, `question_transformed`, `transformation_type`, `transformation_mode`, `transformation_target`, `paraphrase_review: pending`; в режиме `combined` также `transformation_chain`; журнал событий через `--log-file`. |
| `generate_answers.py` | `--input`: JSONL вариантов с уникальными `variant_id`, `paraphrase_review: approved/rejected`; утверждённым нужны `question_id`, `question_original`, `question_transformed`; конфигурация SLM. | `--output`: по строке на утверждённый вариант: `question_id`, `variant_id`, `question_original`, `question_transformed`, `answer_original`, `answer_transformed`, `slm_model` и метаданные. |
| `compute_stability.py` | Обычно `--input`: JSONL ответов с `question_id`, уникальным `variant_id`, обоими вопросами и обоими ответами; конфигурация оценщика. С `--recompute`: сгруппированный JSONL с `question_id` и непустым `variants`, у каждого варианта `answer_equivalence`. | `--output`: JSONL по одной строке на `question_id`, с исходными `question`, `answer`, массивом `variants` (включая `answer_equivalence`), `k`, `N`, `unresolved`, `target`, `review_status: pending`; `target = null` при неясных оценках. `--recompute` пересчитывает числа без модели; `--finalize` оставляет только утверждённые строки с определённым `target`. |
| `check_dataset.py` | `--input`: итоговый JSONL после `--finalize`; пороги `--min-questions`, `--min-variants-per-question`, `--min-total-variants`. | JSON-отчёт в stdout и, если задан `--report`, файл с `passed`, `questions`, `total_variants`, `problems`; при провале код возврата 1. |
| `split_dataset.py` | `--input`: тот же итоговый JSONL; доли `--train`, `--val`, `--test` и `--seed`. | `--output-dir`: `train.jsonl`, `val.jsonl`, `test.jsonl` (исходные строки плюс `split`) и `split_manifest.json` с входным путём, seed, долями и числом строк. |

Итоговая строка для `check_dataset.py` и `split_dataset.py` содержит `question_id`, непустые `question` и `answer`, `review_status: approved`, непустой `variants`. Каждый вариант содержит уникальный `variant_id`, непустые `question` и `answer`, `answer_equivalence: equivalent/non_equivalent`. Проверяется `N = len(variants)`, `k = число equivalent`, `unresolved = 0`, `target = k/N`; для разделения нужны минимум три вопроса и положительные доли с суммой 1.

## Обучение и анализ

| Скрипт | Входные данные | Выходные данные |
|---|---|---|
| `build_embeddings.py` | `--splits-dir`: `train.jsonl`, `val.jsonl`, `test.jsonl` после `split_dataset.py`; использует исходные `question_id`, `question`, `answer` и проверяет всю итоговую запись; модель `intfloat/multilingual-e5-base`, необязательный `--revision`. | `--output-dir`: `{train,val,test}.npz` с `question`, `answer` (`float32`, `[n,768]`) и `question_id`; `{train,val,test}_lengths.jsonl` с длинами и признаками обрезания; `manifest.json` с моделью, revision и хешами. |
| `train_mlp.py` | `--splits-dir`: `train.jsonl`, `val.jsonl`, `test.jsonl`, `split_manifest.json`; `--features-dir`: `train.npz`, `val.npz`, `manifest.json`; `--config`: JSON параметров MLP. | `--output-dir`: `model.pt` (веса), `model.json` (конфигурация, хеши входов, постоянный прогноз), `train_report.json` (история и метрики на `val`). |
| `predict_mlp.py` | `--splits-dir`: `train.jsonl`, `val.jsonl`, `test.jsonl`, `split_manifest.json`; `--features-dir`: NPZ выбранной части и `manifest.json`; `--model-dir`: `model.pt`, `model.json`; для `test` требуется `--final`. | `--output-dir`: `predictions.jsonl` с `question_id`, `split`, `question`, `answer`, `k`, `N`, `target`, `prediction`, `absolute_error`, `constant_prediction`; `predictions.csv` с ID и числовыми полями; `metrics.json`. |
| `analyze_predictions.py` | `--input`: `predictions.jsonl`; `--splits-dir`: `train.jsonl` и JSONL оцениваемой части; необязательный `--features-dir`: `manifest.json` и `{split}_lengths.jsonl`; для `test` требуется `--final`. | `--output-dir`: `analysis.json`, `group_metrics.csv`, `calibration.csv`, `top_errors.jsonl`, `report.md`. |

MLP использует в признаках только исходный вопрос и исходный ответ; варианты нужны для проверки метки и анализа групп.
