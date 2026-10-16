# Полный запуск пайплайна

Все команды выполняются **из корня репозитория** в PowerShell. Установите зависимости через `uv sync` и запустите OpenAI-совместимый сервер моделей на `http://localhost:9090`; конфигурация `configs/model_tools.openai-local.json` сейчас задаёт `qwen3.8-9b-q4` для генератора, фиксированной SLM и оценщика. Проверьте доступность модели по адресу `http://localhost:9090/v1/models` и при необходимости измените имена моделей в конфигурации. Холодный запуск сервера может занять время.

```powershell
uv sync
```

Скрипты записывают UTF-8 JSONL и не перезаписывают существующие выходные файлы. В примерах используются пути `work/...`; для повторного прогона выбирайте новые имена файлов или каталогов. Статусы проверки человек выставляет в JSONL вручную; суффикс `_reviewed` в следующих этапах обозначает отдельную копию после проверки, а исходные вопросы проверяются прямо в `work/questions.jsonl`. Сохраняйте `question_id` и `variant_id` при редактировании, поскольку они связывают этапы.

## 1. Получить исходные вопросы

Сгенерировать 200 кандидатов одной командой, разбив обращения к модели на партии по 20:

```powershell
uv run python scripts/generate_questions.py --config configs/model_tools.openai-local.json --topic "Математика" --count 200 --batch-size 20 --duplicate-check exact --output work/questions.jsonl
```

`--duplicate-check exact` отмечает буквальные повторы в новом файле и, если задан `--existing work/previous_questions.jsonl`, в существующем наборе. **Смысловые дубликаты на этом шаге проверяет человек**: скрипт пока не составляет список похожих пар, а для остальных записей ставит `duplicate_status: unchecked`. `exact` также является режимом по умолчанию; оценщик дубликатов не вызывается. Все новые записи имеют `question_review: pending`: откройте сам `work/questions.jsonl`, проверьте однозначность, условия и дубликаты и замените `pending` на `approved` или `rejected` в каждой строке. Оставшийся `pending` остановит следующий этап; число утверждённых уникальных задач может оказаться меньше 200.

В каждый запрос генератор получает все вопросы из `--existing` и из уже готовых партий текущего запуска; ему задано избегать повторения типов задач и замены только чисел. При `--batch-size 20` контекст обновляется раз в 20 вопросов, а `--batch-size 1` обновляет его перед каждым новым вопросом ценой 200 отдельных запросов. `work/questions.jsonl` создаётся после завершения генерации, поэтому для следующего запуска передайте этот файл через `--existing` и выберите новый путь `--output`. Контекст помогает разнообразию, но не гарантирует отсутствие смысловых дубликатов.

Если модель вернёт обрезанный JSON или неполную партию, скрипт уменьшит размер партии вдвое и повторит запрос; уже полученные вопросы при этом остаются в контексте. Лимит ответа генератора задаётся полем `generator.num_predict` в конфигурации (для локального сервера сейчас 4096). Если ошибка повторяется даже для одного вопроса, проверьте лимит выдачи и размер контекста на сервере; до успешного завершения общий выходной JSONL не записывается.

Если вопросы уже подготовлены в CSV/TSV, вместо генерации импортируйте их:

```powershell
uv run python scripts/import_questions.py --input work/source.csv --question-column question --source pilot --output work/questions.jsonl
```

Для собственной колонки ID используйте `--id-column question_id`, для иного разделителя — `--delimiter`; необязательные `topic` и `language` сохраняются. Импорт проверяет пустые вопросы, повторные ID и буквальные дубликаты внутри входного файла, после чего требуется та же ручная проверка `question_review`.

## 2. Создать и проверить перефразы

Для отдельных способов преобразования используйте `--mode`; `--count` задаёт число новых вариантов **на каждый** утверждённый вопрос, а цели перебираются по кругу. Доступные языки, структуры и стили задаются в `configs/paraphrase_languages.txt`, `configs/paraphrase_structures.txt` и `configs/paraphrase_styles.txt` — по одной цели на строку. Скрипт читает соответствующий файл автоматически. В режимах `translation`, `structure` и `style` запросы идут по одному варианту на цель, поэтому `--batch-size` к ним не применяется.

```powershell
uv run python scripts/generate_paraphrases.py --config configs/model_tools.openai-local.json --input work/questions.jsonl --mode translation --count 3 --output work/paraphrases_translation.jsonl
uv run python scripts/generate_paraphrases.py --config configs/model_tools.openai-local.json --input work/questions.jsonl --mode structure --count 3 --output work/paraphrases_structure.jsonl
uv run python scripts/generate_paraphrases.py --config configs/model_tools.openai-local.json --input work/questions.jsonl --mode style --count 3 --output work/paraphrases_style.jsonl
```

Строки с `#` в файлах целей пропускаются; порядок строк определяет порядок генерации, после конца списка он повторяется. Если нужен другой набор только для одного запуска, его можно передать через `--targets`. В записи сохраняются `transformation_mode`, `transformation_target` и `transformation_type` вида `translation:English`; поле `language` по-прежнему обозначает язык исходного вопроса. При нескольких запусках передавайте предыдущие результаты через `--existing` и выбирайте новый `--output`, чтобы исключить повторы и сохранить уникальные `variant_id`. Перевод и структурное представление особенно важно проверить вручную: модель может изменить смысл или потерять условие.

Каскадный режим применяет структуру, затем стиль к каждой структуре, затем перевод каждого стилевого варианта на каждый язык:

```powershell
uv run python scripts/generate_paraphrases.py --config configs/model_tools.openai-local.json --input work/questions.jsonl --mode combined --count 3 --output work/paraphrases_combined.jsonl
```

Здесь `--count 3` даёт `3 × 3 × 3 = 27` итоговых вариантов на исходный вопрос и 39 обращений к генератору без повторных попыток. Используются первые N строк каждого файла; для каждого этапа нужны как минимум N разных целей. Списки можно переопределить аргументами `--structures`, `--styles` и `--languages`. Итоговые строки содержат `transformation_chain` с текстом каждого промежуточного шага; промежуточные варианты не записываются как отдельные перефразы. Для защиты от случайного слишком большого запуска `--count` в этом режиме ограничен десятью. Поскольку ошибки могут накапливаться на трёх шагах, перед включением в датасет проверяйте и итоговый текст, и всю цепочку.

Прежняя общая генерация доступна без `--mode`:

```powershell
uv run python scripts/generate_paraphrases.py --config configs/model_tools.openai-local.json --input work/questions.jsonl --count 30 --batch-size 5 --output work/paraphrases.jsonl
```

Скрипт обрабатывает только вопросы с `question_review: approved`; `--count 30` означает 30 **новых** кандидатов на каждый такой вопрос. Запросы идут партиями по пять: для каждого исходного вопроса генератор получает только его уже созданные перефразы, а история другого вопроса ему не передаётся. `--batch-size 1` обновляет контекст перед каждым новым перефразом ценой большего числа вызовов. Для продолжения после прежнего запуска укажите `--existing work/paraphrases_old.jsonl` и новый путь `--output`: предыдущие варианты попадут в контекст и их `variant_id` не повторятся, а выходной файл будет содержать только новые записи. Перед следующим этапом объедините старые и новые строки, если нужны оба набора. Скрипт отсеивает буквальные повторы исходного вопроса и уже созданных вариантов; похожие формулировки и сохранение условий требуют ручной проверки. В копии `work/paraphrases_reviewed.jsonl` замените `paraphrase_review: pending` на `approved` или `rejected`.

Если модель вернёт повторы, меньше вариантов, чем запрошено, или отдельные записи без нужных полей, скрипт сохранит пригодные уникальные варианты и запросит недостающее число. Если модель вернёт **больше** вариантов, скрипт просмотрит их все и возьмёт нужное число новых: повтор в начале ответа не закроет путь к уникальным вариантам дальше. При некорректном JSON он повторит запрос с меньшей партией; после пяти подряд ответов без нового пригодного варианта остановится с ошибкой. Каждый запрос немедленно записывает в `work/paraphrases.log.jsonl` события `llm_request` с **полным API payload** (модель, оба сообщения, контекст предыдущих перефразов, temperature, seed и лимит токенов), `llm_response` с полным текстом ответа и причиной завершения, а затем `batch_result` с числами принятых, повторных и непригодных вариантов. События одного запроса связаны полем `request_id`, а разных запусков — `run_id`. В терминале путь к журналу показывается в начале запуска; другой путь можно задать через `--log-file`. Журнал дописывается и сохраняется после ошибки, итоговый `work/paraphrases.jsonl` появляется только после завершения всех вопросов.

Последние события после отказа можно посмотреть командой `Get-Content work/paraphrases.log.jsonl -Tail 10`.


## 3. Получить ответы фиксированной SLM

```powershell
uv run python scripts/generate_answers.py --config configs/model_tools.openai-local.json --input work/paraphrases_reviewed.jsonl --output work/answers.jsonl
```

Скрипт получает один исходный ответ на вопрос и ответы на его принятые перефразы; в строках сохраняются `answer_original`, `answer_transformed` и модель SLM. Проверьте, что ответы получены и пары относятся к нужным вопросам, прежде чем считать метку.

## 4. Сравнить смысл ответов и вычислить k/N

```powershell
uv run python scripts/compute_stability.py --config configs/model_tools.openai-local.json --input work/answers.jsonl --output work/dataset.jsonl
```

Оценщик сравнивает исходный ответ с ответом на каждый перефраз. Для каждого исходного вопроса `N` — число принятых перефразов с ответами, `k` — число их ответов со статусом `equivalent`, `target = k/N`; исходный вопрос в `N` не входит. Если оценщик вернул `unclear` или `insufficient_context`, `target` остаётся `null`. Проверьте строки и статусы внутри `variants`, исправьте спорные `answer_equivalence`, выставьте `review_status: approved` или `rejected` в копии `work/dataset_reviewed.jsonl`, затем пересчитайте метки без обращения к модели и выпустите только утверждённые строки:

```powershell
uv run python scripts/compute_stability.py --recompute --finalize --input work/dataset_reviewed.jsonl --output work/dataset_final.jsonl
```

## 5. Проверить набор и разделить его

Числа ниже служат примером: задайте минимумы и пропорции под свой эксперимент. Проверка вернёт ошибку, если требования не выполнены; для трёх непустых частей нужны как минимум три утверждённых вопроса.

```powershell
uv run python scripts/check_dataset.py --input work/dataset_final.jsonl --min-questions 100 --min-variants-per-question 3 --min-total-variants 300 --report work/check_report.json
uv run python scripts/split_dataset.py --input work/dataset_final.jsonl --output-dir work/splits --train 0.8 --val 0.1 --test 0.1 --seed 42
```

Разбиение сохраняет все перефразы одного вопроса в одной части и создаёт `train.jsonl`, `val.jsonl`, `test.jsonl` и `split_manifest.json` в `work/splits`.

## 7. Получить эмбеддинги, обучить MLP и проверить прогнозы

Для `multilingual-e5-base` начните с `--batch-size 16` и подберите устройство и размер партии по доступной памяти GPU. Параметры MLP задаются в `configs/mlp.example.json`; выбирайте их по `val`.

```powershell
uv run python scripts/build_embeddings.py --splits-dir work/splits --output-dir work/features --device cuda:0 --batch-size 16
uv run python scripts/train_mlp.py --splits-dir work/splits --features-dir work/features --config configs/mlp.example.json --output-dir work/model-001 --device cuda:0
uv run python scripts/predict_mlp.py --splits-dir work/splits --features-dir work/features --model-dir work/model-001 --split val --output-dir work/val-001 --device cuda:0
uv run python scripts/analyze_predictions.py --input work/val-001/predictions.jsonl --splits-dir work/splits --features-dir work/features --output-dir work/val-analysis-001
```

`build_embeddings.py` сохраняет векторы исходных вопроса и ответа в `work/features/{train,val,test}.npz`, а `train_mlp.py` — модель и отчёт обучения. `predict_mlp.py` создаёт `predictions.jsonl`, CSV и метрики; `analyze_predictions.py` создаёт отчёт, метрики по группам и `top_errors.jsonl` для ручного разбора. После выбора модели и настроек по `val` итоговый прогноз на отложенном `test` запускается отдельно:

```powershell
uv run python scripts/predict_mlp.py --splits-dir work/splits --features-dir work/features --model-dir work/model-001 --split test --final --output-dir work/test-final-001 --device cuda:0
```

Подробности контрактов и форматов: [`docs/dataset_cli.md`](../docs/dataset_cli.md), [`docs/mlp_pipeline.md`](../docs/mlp_pipeline.md), [`docs/pipeline_contracts.md`](../docs/pipeline_contracts.md). Низкоуровневые единичные вызовы моделей доступны через `python -m slm_answer_stability.model_tools`; для полного файлового цикла используйте команды выше.
