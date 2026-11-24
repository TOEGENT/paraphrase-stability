from predictors import Dummy
import config

output_dir = config.PREDICTIONS_DIR 


lyr_train_path = config.DATASET_SPLIT_DIR / "LYR" / "train.parquet"
lyr_val_path = config.DATASET_SPLIT_DIR / "LYR" / "val.parquet"


ag_train_path = config.DATASET_SPLIT_DIR / "ag_news" / "train.parquet"
ag_val_path = config.DATASET_SPLIT_DIR / "ag_news" / "val.parquet"

carer_train_path = config.DATASET_SPLIT_DIR / "CARER" / "train.parquet"
carer_val_path = config.DATASET_SPLIT_DIR / "CARER" / "val.parquet"


output_dir = config.PREDICTIONS_DIR 

Dummy.train_dummy_standardized(lyr_train_path,lyr_val_path,output_dir / "Dummy" /"prediction.parquet")

