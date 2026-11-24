from pathlib import Path




LYR_LABEL_MAP = {
    0: "Negative", 1: "Positive", 
}
AG_NEWS_LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Buisness",
    3: "Sci/Tech"
}
CARER_LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

DATASETS_SCHEMA = {'text': 'str', 'label': 'int64'}

DATASETS_IN_DIR = Path("src\\datasets_in")
DATASETS_OUT_DIR = Path("src\\datasets_out")

DATASET_OUTPUT_NAME = "raw_stability"

#DIRS
DATASET_LYR_DIR = Path("src\\datasets_in\\LYR")
DATASET_AG_NEWS_DIR = Path("src\\datasets_in\\ag_news")
DATASET_CARER_DIR = Path("src\\datasets_in\\CARER")

#NAMES
LYR_DATASET_TRAIN_INPUT_NAME = "LYR_train_from_orig" 
LYR_DATASET_VAL_INPUT_NAME = "LYR_val_from_orig"

AG_NEWS_DATASET_TRAIN_INPUT_NAME = "AG_news_train_from_orig"
AG_NEWS_DATASET_VAL_INPUT_NAME = "AG_news_val_from_orig"

CARER_DATASET_TRAIN_INPUT_NAME = "CARER_train"
CARER_DATASET_VAL_INPUT_NAME = "CARER_val"

#DATASET_IN
LYR_TRAIN_PATH = DATASETS_IN_DIR / "LYR" / f"{LYR_DATASET_TRAIN_INPUT_NAME}.parquet"
LYR_VAL_PATH = DATASETS_IN_DIR / "LYR" / f"{LYR_DATASET_VAL_INPUT_NAME}.parquet"

AG_NEWS_TRAIN_PATH = DATASETS_IN_DIR /"ag_news"/f"{AG_NEWS_DATASET_TRAIN_INPUT_NAME}.parquet"
CARER_TRAIN_PATH = DATASETS_IN_DIR /"CARER"/f"{CARER_DATASET_TRAIN_INPUT_NAME}.parquet"

#DATASET_OUT
LYR_JSONL_PATH = DATASETS_OUT_DIR / "LYR" / f"{DATASET_OUTPUT_NAME}.jsonl"
LYR_OUTPUT_DATASET_PATH = DATASETS_OUT_DIR / "LYR" / f"{DATASET_OUTPUT_NAME}.parquet"

AG_NEWS_JSONL_PATH = DATASETS_OUT_DIR / "ag_news" / f"{DATASET_OUTPUT_NAME}.jsonl"
AG_NEWS_OUTPUT_DATASET_PATH = DATASETS_OUT_DIR / "ag_news" / f"{DATASET_OUTPUT_NAME}.parquet"

CARER_JSONL_PATH = DATASETS_OUT_DIR / "CARER" / f"{DATASET_OUTPUT_NAME}.jsonl"
CARER_OUTPUT_DATASET_PATH = DATASETS_OUT_DIR / "CARER" / f"{DATASET_OUTPUT_NAME}.parquet"

#DATASET_SPLIT
DATASET_SPLIT_DIR = Path("src\\datasets_out_split")

#PREDICTIONS
PREDICTIONS_DIR = Path("src\\predictions")
