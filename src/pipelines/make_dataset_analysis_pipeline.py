from utils.for_analys import(
    make_k_diagram,
    make_uniqueK_over_k_bar
)
import config

#СДЕЛАТЬ ДОКУМЕНТАЦИЮ!!!!

def make_dataset_analysis_pipeline(dataset_json_path):
    make_k_diagram.make_k_diagram(dataset_json_path)
    make_uniqueK_over_k_bar.make_uniqueK_over_k_bar(dataset_json_path)

if __name__ == "__main__":
    make_dataset_analysis_pipeline(config.LYR_JSONL_PATH)