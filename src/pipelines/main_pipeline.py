"""
Задача
    принять датасет LYR 
    выдать аналитику предсказаний предикторами множества

    Решение:
        


"""



from pipelines import (
    make_dataset_analysis_pipeline,
    make_dataset_pipeline,
    
)
import config

input_file = config.LYR_TRAIN_PATH
output_folder = config.DATASET_LYR_DIR





make_dataset_pipeline.make_dataset_pipeline(input_file, output_folder)

make_dataset_analysis_pipeline.make_dataset_analysis_pipeline(config.LYR_JSONL_PATH)

make_predictors_pipeline()

make_predictor_analysis_pipeline.make_predictor_analysis_pipeline(input_file, output_folder)
