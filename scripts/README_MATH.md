# Пайплайн для математического датасета

Все команды выполняйте из корня репозитория в PowerShell. Установите зависимости через `uv sync`. Конфигурация `configs/model_tools.openai-local.json` использует OpenAI-совместимый сервер `http://172.16.34.8:8080/v1` и модель `qwen3.5-9b`; проверьте доступность сервера и при необходимости измените конфигурацию. Для каждого повторного запуска выбирайте новые имена выходных файлов: скрипты не перезаписывают существующие.

```powershell
uv sync
```

## 1. Сгенерировать уравнения и проверить варианты

```powershell
uv run python scripts/generate_equation_permutations.py --count 10 --seed 42 --max-variants 100 --output work/equation_variants.jsonl
```

`--count` задаёт число исходных уравнений, `--max-variants` — предел вариантов на одно уравнение (не более 100), `--seed` позволяет повторить генерацию. Скрипт собирает линейные уравнения из знаковых слагаемых, переставляет слагаемые и стороны равенства; он не импортирует готовые задачи DART-Math и не вычисляет ответы. Варианты с перестановкой слагаемых только на одной стороне без обмена сторон пропускаются, поэтому вариантов может быть меньше заданного предела.

Создайте отдельную копию `work/equation_variants_reviewed.jsonl`, проверьте пары `question_original` и `question_transformed` и замените в каждой строке `paraphrase_review: "pending"` на `"approved"` или `"rejected"`. Сохраняйте `question_id` и `variant_id`: они связывают этапы. Исходный вопрос и варианты уже находятся в одном JSONL, поэтому обычная генерация вопросов и перефразов здесь не нужна.

## 2. Получить ответы SLM

```powershell
uv run python scripts/generate_answers.py --config configs/model_tools.openai-local.json --input work/equation_variants_reviewed.jsonl --output work/equation_answers.jsonl
```

Скрипт обрабатывает только утверждённые варианты, получает один ответ на исходное уравнение и отдельный ответ на каждый вариант. В выходных строках есть `answer_original`, `answer_transformed` и `slm_model`. Системная инструкция в конфигурации требует для уравнения одну строку `x = NUMBER`; после изменения инструкции ответы необходимо получить заново.

## 3. Сравнить итоговые значения x

```powershell
uv run python scripts/equivalence_generation.py --input work/equation_answers.jsonl --output work/equation_all_equivalence.jsonl
```

По умолчанию скрипт делает исходный вопрос и **каждый** вариант опорным, сравнивает его ответ со всеми остальными ответами того же уравнения и сохраняет отдельный статус `answer_equivalence` для каждого сравнения в массиве `comparisons`. В терминале `counts` суммирует все направленные сравнения, а число строк выходного файла равно числу опорных пар. Скрипт распознаёт целые, десятичные, `a/b` и LaTeX-дроби для одного `x`; неоднозначный ответ останавливает расчёт для ручной проверки. Совпадение двух неверных ответов всё равно считается эквивалентностью.

## 4. При необходимости проверить правильность каждого ответа

```powershell
uv run python scripts/check_math_answer_correctness.py --input work/equation_answers.jsonl --output work/equation_answers_checked.jsonl
```

Скрипт подставляет извлечённый `x` в исходное и преобразованное уравнения и добавляет `answer_original_correctness` и `answer_transformed_correctness` со статусами `correct`, `incorrect`, `unparseable` или `unsupported`. Он принимает вопрос в формате `Solve for x: $левая часть = правая часть$.` и линейные выражения с поддерживаемым синтаксисом `+`, `-`, скобок, умножения и деления на ненулевую константу; другие форматы и уравнения без единственного решения получают `unsupported`. В терминале исходный ответ для одного `question_id` учитывается один раз, а варианты считаются отдельно. Этот этап не меняет эквивалентность и не записывает эталонное решение.

## 5. Вычислить k/N для каждой опорной пары

Основной режим `compute_k_N.py` принимает попарные оценки после основного режима `equivalence_generation.py` и записывает отдельную метку для **каждой** пары «формулировка — ответ»:

```powershell
uv run python scripts/compute_k_N.py --input work/equation_all_equivalence.jsonl --output work/equation_all_anchors.jsonl
```

Скрипт проверяет полноту попарных оценок и их согласованность с извлечёнными `x`, суммирует `equivalent` для каждого якоря и записывает `k`, `N`, `target`, `source_question_id` и `split`. По умолчанию исходные уравнения делятся на `train/val/test` в пропорции 0,8/0,1/0,1 с seed 42; все якоря одного уравнения остаются вместе. Параметры можно изменить через `--train`, `--val`, `--test` и `--seed`, а прежнее разбиение повторить через `--splits-dir`. Для текущих 6 170 вариантов результат содержит 6 242 опорные строки вместе с 72 исходными.

## 6. При необходимости: прежняя метка только для исходного вопроса

Флаг `--original-only` у обоих скриптов оставляет по одной оценке эквивалентности на вариант и одной групповой метке на исходное уравнение. Если выполнен шаг 4, можно передать `work/equation_answers_checked.jsonl` вместо сырого файла ответов в первый скрипт, так как он сохраняет исходные поля.

```powershell
uv run python scripts/equivalence_generation.py --original-only --input work/equation_answers.jsonl --output work/equation_answers_original_verified.jsonl
uv run python scripts/compute_k_N.py --original-only --input work/equation_answers_original_verified.jsonl --output work/equation_dataset.jsonl
```

Для каждого исходного вопроса `N` — число вариантов с ответами, `k` — число вариантов со статусом `equivalent`, `target = k/N`; при `unclear` или `insufficient_context` значение `target` остаётся `null`. Проверьте статусы внутри `variants`, при необходимости исправьте их и в отдельной копии `work/equation_dataset_reviewed.jsonl` выставьте для каждой группы `review_status: "approved"` или `"rejected"`. Затем пересчитайте метки без вызова модели и выпустите утверждённые строки:

```powershell
uv run python scripts/compute_k_N.py --original-only --recompute --finalize --input work/equation_dataset_reviewed.jsonl --output work/equation_dataset_final.jsonl
```

Для прежнего группового датасета пороги проверки ниже служат примером; подберите их под фактический размер набора. Для трёх непустых частей нужны минимум три утверждённых исходных вопроса.

```powershell
uv run python scripts/check_dataset.py --input work/equation_dataset_final.jsonl --min-questions 3 --min-variants-per-question 3 --min-total-variants 9 --report work/equation_check_report.json
uv run python scripts/split_dataset.py --input work/equation_dataset_final.jsonl --output-dir work/equation_splits --train 0.8 --val 0.1 --test 0.1 --seed 42
```

Разбиение сохраняет все варианты одного вопроса в одной части и создаёт `train.jsonl`, `val.jsonl`, `test.jsonl` и `split_manifest.json`.

## 7. Получить эмбеддинги, обучить MLP и проверить прогнозы

Для набора **всех опорных пар** `build_embeddings.py` создаёт `train.jsonl`, `val.jsonl`, `test.jsonl` и `split_manifest.json` в `--splits-dir`, проверяет отсутствие пересечений по `source_question_id` и сохраняет эмбеддинги в `--output-dir`. Для запуска на CPU используйте последовательность ниже; на доступном GPU можно заменить `cpu` на `cuda:0` и подобрать `--batch-size` по памяти.

```powershell
uv run python scripts/build_embeddings.py --anchors-input work/equation_all_anchors.jsonl --splits-dir work/equation_anchor_splits --output-dir work/equation_anchor_features --device cpu --batch-size 16
uv run python scripts/train_mlp.py --splits-dir work/equation_anchor_splits --features-dir work/equation_anchor_features --config configs/mlp.example.json --output-dir work/equation_anchor_model-001 --device cpu
uv run python scripts/predict_mlp.py --splits-dir work/equation_anchor_splits --features-dir work/equation_anchor_features --model-dir work/equation_anchor_model-001 --split val --output-dir work/equation_anchor_val-001 --device cpu
```

После выбора модели по `val` выполните финальный прогноз на `test`, сохраняя **те же** каталоги разбиения, признаков и модели:

```powershell
uv run python scripts/predict_mlp.py --splits-dir work/equation_anchor_splits --features-dir work/equation_anchor_features --model-dir work/equation_anchor_model-001 --split test --final --output-dir work/equation_anchor_test-final-001 --device cpu
```

Если каталоги `work/equation_anchor_splits`, `work/equation_anchor_features` и `work/equation_anchor_model-001` уже созданы, первые две команды повторять не нужно. В текущем запуске файл меток называется `work/equation_all_anchors_from_pairs.jsonl`; он использовался при подготовке этих каталогов и не требует переименования. Для каждого нового запуска выбирайте новые имена выходных файлов и каталогов: скрипты не перезаписывают существующие. Конфигурация `configs/mlp.example.json` с `loss_weighting: equal` даёт одинаковый вес каждой опорной строке, поэтому уравнения с 100 вариантами имеют больший суммарный вес, чем уравнения с 61 или 6 вариантами. Все якоря одного уравнения зависимы, поэтому число строк `test` не равно числу независимых тестовых задач.

Для прежнего набора с **одним якорем на исходный вопрос** используйте отдельные каталоги и следующие команды:

```powershell
uv run python scripts/build_embeddings.py --splits-dir work/equation_splits --output-dir work/equation_features --device cpu --batch-size 16
uv run python scripts/train_mlp.py --splits-dir work/equation_splits --features-dir work/equation_features --config configs/mlp.example.json --output-dir work/equation_model-001 --device cpu
uv run python scripts/predict_mlp.py --splits-dir work/equation_splits --features-dir work/equation_features --model-dir work/equation_model-001 --split val --output-dir work/equation_val-001 --device cpu
uv run python scripts/analyze_predictions.py --input work/equation_val-001/predictions.jsonl --splits-dir work/equation_splits --features-dir work/equation_features --output-dir work/equation_val-analysis-001
```

Только для этой групповой схемы после выбора настроек по `val` проверьте отложенную часть `test`:

```powershell
uv run python scripts/predict_mlp.py --splits-dir work/equation_splits --features-dir work/equation_features --model-dir work/equation_model-001 --split test --final --output-dir work/equation_test-final-001 --device cpu
```

`analyze_predictions.py` пока ожидает групповые записи с массивом `variants` и не принимает компактный набор всех якорей.

Форматы и ограничения подробнее описаны в [контрактах пайплайна](../docs/pipeline_contracts.md), [документации CLI](../docs/dataset_cli.md) и [руководстве по MLP](../docs/mlp_pipeline.md).
