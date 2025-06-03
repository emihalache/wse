import pandas as pd

class Preprocessor:
    def preprocess(self, df):
        new_df = df.drop_duplicates()
        return new_df