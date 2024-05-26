from dspy.datasets.dataset import Dataset
import pandas as pd

class PrbSolvDataset(Dataset):
    def __init__(self, file_path, **kwargs) -> None:
        super().__init__()
        df = pd.read_excel(file_path)
        df.loc[1:,"problem"] = "problem statement : " + df["problem"] + "\n" + "Can you generate an algorithm for the given problem statement ?" 
        df.loc[1:,"python"] = "code generated with explanations: " + df["python"]
        self._train = df.loc[1:30, ["problem", "python"]].to_dict(orient='records')
        self._dev = df.loc[30:, ["problem", "python"]].to_dict(orient='records')

if __name__ == '__main__':
    dataset = PrbSolvDataset('problem_solving/dataset.xlsx')
    print(dataset.train[:3])