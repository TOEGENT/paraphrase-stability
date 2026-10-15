# Вызов моделей для подготовки датасета

Первый работающий интерфейс — `slm_answer_stability.model_tools`. Он обращается к локальному [Ollama Chat API](https://github.com/ollama/ollama/blob/main/docs/api.md) через стандартную библиотеку Python и задаёт модели отдельно для трёх ролей: фиксированная SLM, генератор и оценщик. Пример настройки — [`configs/model_tools.example.json`](../configs/model_tools.example.json); замените имена моделей и настройки под ваш запуск. Для пробного запуска все три роли используют одну установленную модель, но в исследовательском датасете версии моделей и промптов следует фиксировать отдельно.

Из корня репозитория после `pip install -e .`:

```bash
python -m slm_answer_stability.model_tools questions --topic "elementary arithmetic" --count 2
python -m slm_answer_stability.model_tools paraphrases --question "What is 2+2?" --count 2
python -m slm_answer_stability.model_tools answer --question "What is 2+2?"
python -m slm_answer_stability.model_tools judge-question --question "What is 2+2?" --rewritten-question "What is two plus two?"
python -m slm_answer_stability.model_tools judge-answer --question "What is 2+2?" --answer "4" --rewritten-answer "Four"
```

Каждая команда печатает один JSON-объект в stdout; ошибки идут в stderr и завершаются ненулевым кодом. `questions` создаёт **синтетические** исходные задачи, `paraphrases` — варианты существующего вопроса, `answer` вызывает фиксированную SLM, а две команды `judge-*` независимо проверяют смысл вопроса и ответа. Для иной конфигурации добавьте `--config путь/к/файлу.json` перед названием команды.

Этот CLI — слой доступа к моделям. Его короткие запросы задают JSON-формат ответа CLI; файлы в `prompts/` задают формат JSONL исследовательского протокола и напрямую сюда не подставляются. Файловые этапы с ручными проверками и расчётом `k/N` описаны в [`docs/dataset_cli.md`](../docs/dataset_cli.md), а команды `build_embeddings.py`, `train_mlp.py` и `predict_mlp.py` — в [`docs/mlp_pipeline.md`](../docs/mlp_pipeline.md). На Windows с установленным Ollama проверьте доступные модели командой `ollama list`; нужные модели должны быть загружены до запуска CLI.
