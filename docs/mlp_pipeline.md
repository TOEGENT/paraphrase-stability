# Прогон MLP

Вход — каталог `splits` после `split_dataset.py`: `train.jsonl`, `val.jsonl`, `test.jsonl` и `split_manifest.json`. Каждый JSONL содержит утверждённые исходные вопрос и ответ, `k`, `N`, `target` и вложенные варианты; в MLP попадают только исходный вопрос и ответ.

Установите зависимости: `pip install -e ".[mlp]"`. Команда автоматически определит текущий commit hash `multilingual-e5-base` и сохранит его в манифесте; для точного повторения запуска передайте сохранённое значение в `--revision`. Начните с небольшого батча и подберите его по фактической памяти видеокарты; MLP после вычисления эмбеддингов можно обучать даже на CPU.

```bash
python scripts/build_embeddings.py --splits-dir work/splits --output-dir work/features --device cuda:0 --batch-size 16
python scripts/train_mlp.py --splits-dir work/splits --features-dir work/features --config configs/mlp.example.json --output-dir work/model-001 --device cuda:0
python scripts/predict_mlp.py --splits-dir work/splits --features-dir work/features --model-dir work/model-001 --split val --output-dir work/val-001 --device cuda:0
```

`features/{train,val,test}.npz` содержит `question` и `answer` (`float32`, `[n,768]`) и `question_id`; `features/manifest.json` фиксирует модель, revision, префикс `query: `, длину текста, число обрезаний и SHA-256 исходных JSONL и NPZ. Файл `model.pt` содержит веса PyTorch, `model.json` — конфигурацию и контрольные суммы, `train_report.json` — историю обучения, MAE/RMSE MLP и постоянного прогноза на `val`. Скрипт прогнозов создаёт `predictions.csv` и `metrics.json`; время получения эмбеддингов в них не включено.

В `configs/mlp.example.json` по умолчанию `loss_weighting: equal`: каждая задача имеет одинаковый вес в функции потерь. Для биномиального веса поставьте `loss_weighting: binomial`; поле `max_n_weight` может ограничить влияние задач с большим `N`. Это отдельные эксперименты: выбирать настройку следует только по `val`, а не по `test`.

После фиксации модели и настроек итоговую оценку запускают явно:

```bash
python scripts/predict_mlp.py --splits-dir work/splits --features-dir work/features --model-dir work/model-001 --split test --final --output-dir work/test-final-001 --device cuda:0
```

Каталоги выходов должны быть пустыми; скрипты не перезаписывают прежние результаты. Проверка совпадения `question_id`, размеров и хешей предотвращает незаметное смешивание данных разных запусков. Маленький технический пример проверяет исполнимость цепочки, но его метрики не оценивают качество на новых задачах.
