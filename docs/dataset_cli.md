# Запуск скриптов датасета

Поля входа и выхода каждого скрипта перечислены в [pipeline_contracts.md](pipeline_contracts.md). Команды выполняются из корня репозитория после `uv sync`; вызовы моделей требуют подходящего `--config` и запущенного сервера.

```powershell
uv run python scripts/import_questions.py --input work/source.csv --question-column question --source pilot --output work/questions.jsonl
# Или вместо импорта:
uv run python scripts/generate_questions.py --config configs/model_tools.openai-local.json --topic "Математика" --count 200 --output work/questions.jsonl
```

В `work/questions.jsonl` выставьте `question_review: approved/rejected` для каждой строки.

```powershell
uv run python scripts/generate_paraphrases.py --config configs/model_tools.openai-local.json --input work/questions.jsonl --count 30 --output work/paraphrases.jsonl
```

В копии `work/paraphrases_reviewed.jsonl` выставьте `paraphrase_review: approved/rejected`, сохранив `question_id` и `variant_id`.

```powershell
uv run python scripts/generate_answers.py --config configs/model_tools.openai-local.json --input work/paraphrases_reviewed.jsonl --output work/answers.jsonl
uv run python scripts/compute_stability.py --config configs/model_tools.openai-local.json --input work/answers.jsonl --output work/dataset.jsonl
```

В копии `work/dataset_reviewed.jsonl` проверьте `variants[].answer_equivalence` и выставьте `review_status: approved/rejected`; неясные оценки исправьте до выпуска.

```powershell
uv run python scripts/compute_stability.py --recompute --finalize --input work/dataset_reviewed.jsonl --output work/dataset_final.jsonl
uv run python scripts/check_dataset.py --input work/dataset_final.jsonl --min-questions 100 --min-variants-per-question 3 --min-total-variants 300 --report work/check_report.json
uv run python scripts/split_dataset.py --input work/dataset_final.jsonl --output-dir work/splits --train 0.8 --val 0.1 --test 0.1 --seed 42
```

Пороги в примере задаются под конкретный набор. Каждый выходной файл должен иметь новый путь. Режимы перефразирования описаны в [scripts/README.md](../scripts/README.md).
