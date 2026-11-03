# Роль 2: прогон датасета

## Что происходит

Человек этой роли берёт готовые исходные задачи и перефразы, получает ответы одной и той же SLM, сравнивает ответы и собирает метку устойчивости `k/N`. Сначала нужен маленький прогон, чтобы проверить формат данных и оценщика ответов; затем тот же маршрут можно запускать на большом объёме.

```text
исходные задачи → варианты → ответы SLM → сравнение ответов → k/N → проверка и разбиение
```

## Что уже можно запустить

Для линейных задач есть первые два шага. Из корня проекта:

```powershell
uv run python scripts/generate_math_tasks.py --topic algebra.linear_1d --count 20 --seed 42 --format structured --output work/server_pilot_sources.jsonl
uv run python scripts/generate_linear_equation_variants.py --input work/server_pilot_sources.jsonl --variants-per-question 10 --seed 42 --output work/server_pilot_variants.jsonl
```

Ожидаются 20 исходных задач и 200 вариантов; числа следуют из параметров, а не из уже лежащего в `work` результата. Сейчас в `work` **нет файлов этого серверного пилота**, поэтому здесь не приведён выдуманный JSON-вывод. Готовые варианты имеют `paraphrase_review: "pending"`; перед `generate_answers.py` человек должен проверить пары и выставить `approved` или `rejected`.

## Проблема и задача

Скрипты ответов и сборки меток существуют: `generate_answers.py`, `equivalence_generation.py`, `equivalence_generation_ai.py`, `compute_k_N.py`, `check_dataset.py`, `split_dataset.py`. Но полный прогон **новой математической цепочки** до финального JSONL в рабочей папке не подтверждён. Старый `equivalence_generation.py` рассчитан на числовое `x`, тогда как Mathematics Dataset даёт другие имена переменных, системы и символьные ответы. Человеку этой роли нужно на пилоте выбрать и проверить подходящий способ сравнения ответов, записать все команды и числа строк, а после успешного пилота повторить прогон на сервере. Подробный план: `scripts/README/README_SERVER_DATASET_TASK.md`.
