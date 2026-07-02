import pandas as pd


def load_filtered_dataset(file_path: str, length: bool) -> pd.DataFrame:
    """
    Load CSV dataset and filter sequences longer than 2000 residues.

    Args:
        file_path (str): Path to CSV file.
        length (bool): Whether to apply length filtering (20–2000 residues).

    Returns:
        pd.DataFrame: Filtered dataframe with 'sequence' and 'labels'.
    """
    df = pd.read_csv(file_path)
    if not length:
        return df[["ProteinID", "sequence", "Tm"]].rename(columns={'Tm': 'labels'}).reset_index(drop=True)
    df = df[(df.Length <= 2000) & (df.Length >= 20)][["ProteinID", "sequence", "Tm"]].rename(columns={'Tm': 'labels'})
    return df
