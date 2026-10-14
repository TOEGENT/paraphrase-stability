# Scripts (planned)

Ниже — предполагаемые команды запуска этапов (заглушки для будущей реализации):

```bash
python -m slm_answer_stability.data.prepare --config configs/pilot.yaml
python -m slm_answer_stability.generation.run --config configs/pilot.yaml
python -m slm_answer_stability.annotation.run --config configs/pilot.yaml
python -m slm_answer_stability.models.train --config configs/pilot.yaml
python -m slm_answer_stability.evaluation.run --config configs/pilot.yaml
```

Ожидаемый порядок: `data -> generation -> annotation -> models -> evaluation`.
