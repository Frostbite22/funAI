import pandas as pd 
from dspy.datasets.dataset import Dataset

class CSVDataset(Dataset):
    def __init__(self, file_path, **kwargs) -> None:
        super().__init__()

        df = pd.read_excel(file_path)
        self._train = df.iloc[1:18].to_dict(orient='records')
        self._dev = df.iloc[18:].to_dict(orient='records')



if __name__ == '__main__':
    dataset = CSVDataset('rag_data.xlsx')
    print(dataset.train[:3])

