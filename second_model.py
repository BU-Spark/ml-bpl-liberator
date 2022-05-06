from text_classification.lbl2vec_model import run_lbl2vec
from colorama import Fore, init
init(autoreset=True)

FINAL_DATA_OUTPUT = "data/JSON_outputs/final_data.json"

print(Fore.CYAN, "running the text classification algorithm...")
run_lbl2vec(input_data_path="data/JSON_outputs/data.json", output_data_path=FINAL_DATA_OUTPUT)

print(Fore.GREEN + "Complete, wrote data.json to " + FINAL_DATA_OUTPUT)