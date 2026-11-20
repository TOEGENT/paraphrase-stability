import pandas as pd
from pathlib import Path

# Импортируем все компоненты из utils
from utils import (
    split_dataset
)
from utils import convert_jsonl_to_parquet
from utils import generate_llm_inputs
from utils import llm_response_generator
from utils import save_results_to_jsonl

def make_dataset_pipeline(input_path: Path, output_dir: Path):
    if (Path(output_dir) / Path("raw_stability.parquet")).exists():
        raise FileExistsError(
            f"Файл {input_path} уже существует! "
            "Удалите его вручную, если хотите перегенерировать датасет."
        )
    else:
        df_raw = pd.read_parquet(input_path)


    input_iter = generate_llm_inputs.generate_llm_inputs(df_raw)
    llm_iter = llm_response_generator.llm_response_generator(input_iter)

    raw_json_path = output_dir / "raw_stability.jsonl"
    raw_parquet_path = output_dir / "raw_stability.parquet"

    save_results_to_jsonl.save_results_to_jsonl(llm_iter, raw_json_path)
    convert_jsonl_to_parquet.convert_jsonl_to_parquet(raw_json_path,raw_parquet_path)
    split_dataset.split_dataset(str(raw_parquet_path), str(output_dir))
    
    
if __name__ == "__main__":
    input_file = Path("src/datasets/LYR/LYR_train_from_orig.parquet")
    output_folder = Path("src/datasets/LYR")
    make_dataset_pipeline(input_file, output_folder)
