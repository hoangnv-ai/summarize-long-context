import unicodedata
import ast
import json

def str_to_dict(s: str) -> dict:
    return ast.literal_eval(s)
# ---------------------
# Helper bỏ dấu tiếng Việt
# ---------------------
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def load_menu():
    with open('./menu.json', 'r') as file:
        menu = json.load(file)
    return menu

def load_prompt():
    with open("./system_prompt.txt", "r") as f:
        prompt = f.read().strip() 
    return prompt