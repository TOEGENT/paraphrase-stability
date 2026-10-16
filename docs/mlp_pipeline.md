# Запуск MLP и анализа прогнозов

Входные поля и выходные файлы перечислены в [pipeline_contracts.md](pipeline_contracts.md). Команды выполняются из корня репозитория после `uv sync`; нужны готовые `work/splits/{train,val,test}.jsonl` от `split_dataset.py`.

```powershell
uv run python scripts/build_embeddings.py --splits-dir work/splits --output-dir work/features --device cuda:0 --batch-size 16
uv run python scripts/train_mlp.py --splits-dir work/splits --features-dir work/features --config configs/mlp.example.json --output-dir work/model-001 --device cuda:0
uv run python scripts/predict_mlp.py --splits-dir work/splits --features-dir work/features --model-dir work/model-001 --split val --output-dir work/val-001 --device cuda:0
uv run python scripts/analyze_predictions.py --input work/val-001/predictions.jsonl --splits-dir work/splits --features-dir work/features --output-dir work/val-analysis-001
```

После выбора модели по `val` итоговый прогноз и анализ `test` запускаются с `--final`:

```powershell
uv run python scripts/predict_mlp.py --splits-dir work/splits --features-dir work/features --model-dir work/model-001 --split test --final --output-dir work/test-final-001 --device cuda:0
uv run python scripts/analyze_predictions.py --input work/test-final-001/predictions.jsonl --splits-dir work/splits --features-dir work/features --output-dir work/test-analysis-001 --final
```

Для повторяемого вычисления эмбеддингов передайте `--revision` с 40-символьным commit hash модели из `work/features/manifest.json`. Каталоги `--output-dir` должны быть пустыми; скрипты проверяют соответствие ID и хешей входов.
