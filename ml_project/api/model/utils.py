import json
import pandas as pd

def load_data():
    with open('ml_project/data/problem_solutions.json', 'r') as f:
        problem_solutions = json.load(f)
    df = pd.DataFrame(problem_solutions)
    return df
