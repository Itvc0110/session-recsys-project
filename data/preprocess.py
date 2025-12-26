import pandas as pd

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values(by=['user_id', 'timestamp']).drop_duplicates(subset=['user_id', 'item_id'], keep='last')

def sort_by_timestamp(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values(by=['user_id', 'timestamp'])

def apply_5core_filtering(data: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    while True:
        user_counts = data['user_id'].value_counts()
        item_counts = data['item_id'].value_counts()
        prev_len = len(data)
        data = data[data['user_id'].isin(user_counts[user_counts >= min_interactions].index)]
        data = data[data['item_id'].isin(item_counts[item_counts >= min_interactions].index)]
        if len(data) == prev_len:
            break
    return data
