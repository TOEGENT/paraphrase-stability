# Роль 4: MLP и обучаемость

## Что происходит

После роли 2 у каждого вопроса должна быть метка `target = k/N`: какая доля перефраз сохранила ответ модели. Эта роль проверяет, может ли MLP предсказывать `target` по признакам вопроса и ответа, и сравнивает её с простым прогнозом среднего значения.

```text
проверенный датасет → train/val/test → эмбеддинги → MLP → прогноз на val → сравнение с baseline
```

## Какие скрипты уже есть

```powershell
uv run python scripts/split_dataset.py --input work/dataset_final.jsonl --output-dir work/splits --train 0.8 --val 0.1 --test 0.1 --seed 42
uv run python scripts/build_embeddings.py --splits-dir work/splits --output-dir work/features --device cuda:0 --batch-size 16
uv run python scripts/train_mlp.py --splits-dir work/splits --features-dir work/features --config configs/mlp.example.json --output-dir work/model-001 --device cuda:0
uv run python scripts/predict_mlp.py --splits-dir work/splits --features-dir work/features --model-dir work/model-001 --split val --output-dir work/val-001 --device cuda:0
```

Эти команды показывают существующий маршрут, **не состоявшийся запуск**: в текущем `work` нет `dataset_final.jsonl`, `splits`, `features` и результатов MLP. Когда вход будет готов, скрипты создадут `work/splits/{train,val,test}.jsonl`, файлы признаков в `work/features`, модель и отчёт в `work/model-001`, а также `work/val-001/predictions.jsonl` и `metrics.json`. До появления входа показывать численные метрики как факт нельзя.

## Проблема и задача

Код MLP существует, но подтверждённого baseline на новом датасете нет. Сначала получить от роли 2 проверенный финальный JSONL и убедиться, что варианты одного исходного вопроса находятся в одной части train/val/test. Затем сравнить константный baseline и текущую MLP на **одном и том же** разбиении, улучшать модель по `val` и только один раз проверить выбранную конфигурацию на `test`. Подробный план: `scripts/README/README_MLP_TASK.md`.
