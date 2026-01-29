from pathlib import Path
import pandas as pd
from utils.paths import CACHE_PATH


class Sentiment:
    POS = "pos"
    NEG = "neg"


def _load_data(sentiment: str, dataset_path: Path):
    sentiment_value = 1 if sentiment == Sentiment.POS else 0
    files = []
    path_to_data = dataset_path / sentiment
    file_paths = path_to_data.glob("*.txt")
    for file_path in file_paths:
        with open(file=file_path) as text_file:
            files.append(text_file.read())
    sentiment_list = [sentiment_value] * len(files)
    return files, sentiment_list


def load_dataframe(sentiment: str, dataset_path: Path):
    columns = ["text", "sentiment_value"]
    df = pd.DataFrame(
        dict(zip(columns, _load_data(sentiment=sentiment, dataset_path=dataset_path))))
    return df


def load_test_dataframe():
    data_path = CACHE_PATH / "sentiment_embeddings"
    raw_data_path = data_path / "raw" / "aclImdb"
    test_directory_path = raw_data_path / "test"
    df_pos = load_dataframe(Sentiment.POS, test_directory_path)
    df_neg = load_dataframe(Sentiment.NEG, test_directory_path)
    df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    return df
