from pathlib import Path

DATASETS_IN_DIR = Path("src\\datasets_in")
DATASETS_OUT_DIR = Path("src\\datasets_out")

DATASET_LYR_DIR = Path("src\\datasets_in\\LYR")
DATASET_OUTPUT_NAME = "raw_stability"

#датасеты самого LYR
LYR_DATASET_TRAIN_INPUT_NAME = "LYR_train_from_orig" 
LYR_DATASET_VAL_INPUT_NAME = "LYR_val_from_orig"
LYR_TRAIN_PATH = DATASETS_IN_DIR / "LYR" / f"{LYR_DATASET_TRAIN_INPUT_NAME}.parquet"
LYR_VAL_PATH = DATASETS_IN_DIR / "LYR" / f"{LYR_DATASET_VAL_INPUT_NAME}.parquet"

#будущие датасеты
LYR_JSONL_PATH = DATASETS_OUT_DIR / "LYR" / f"{DATASET_OUTPUT_NAME}.jsonl"
LYR_OUTPUT_DATASET_PATH = DATASETS_OUT_DIR / "LYR" / f"{DATASET_OUTPUT_NAME}.parquet"

