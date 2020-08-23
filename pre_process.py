from pathlib import Path
from dask import dataframe
from pandas import concat, Index


class PreProcess:

    def __init__(self, columns: dict, num_workers: int):
        self.cols: dict = columns
        self.workers: int = num_workers

    def load_data(self, path: str):
        return dataframe.read_csv(
            urlpath=path,
            blocksize=int(Path(path).stat().st_size / self.workers)
        )

    def impute_nulls(self, data: dataframe):
        # Impute by mean
        data[self.cols["CONTINUOUS"]] = data[self.cols["CONTINUOUS"]].fillna(
            data[self.cols["CONTINUOUS"]].mean(axis=0, skipna=True)
            .compute(num_workers=self.workers), axis=0)

        # Impute by mode
        cat_cols: list = self.cols["CATEGORICAL"]["STRING"] +\
            self.cols["CATEGORICAL"]["NUMERIC"]
        col_modes = data[cat_cols].mode(
            dropna=True).compute(num_workers=self.workers)
        for col in cat_cols:
            data[col] = data[col].fillna(col_modes[col].iloc[0], axis=0)

        data = data.dropna(how="any")
        return data

    def remove_outliers(self, data: dataframe, threshold: float):

        data = data.compute(num_workers=self.workers)
        stats: dict = {
            "mean": data[self.cols["CONTINUOUS"]].mean(axis=0),
            "std_dev": data[self.cols["CONTINUOUS"]].std(axis=0)
        }

        z_cols = list(map(lambda col: "z" + col, self.cols["CONTINUOUS"]))
        zdata = data[self.cols["CONTINUOUS"]].apply(lambda col: (
            col - stats["mean"][col.name])/(stats["std_dev"][col.name]),
            axis=0)
        zdata.columns = z_cols

        data = concat([data, zdata], axis=1)
        for z_col in z_cols:
            data = data[data[z_col].between(-1 * threshold, threshold)]

        return dataframe.from_pandas(
            data.drop(columns=z_cols).reset_index(drop=True),
            npartitions=self.workers)

    def encode_data(self, data: dataframe, is_train: bool = True,
                    encoding=None):
        if encoding is None:
            encoding = dict()

        for col in self.cols["CATEGORICAL"]["STRING"]:
            if is_train:
                encoding[col] = dict()
                unique_col = data[col].unique().compute(
                    num_workers=self.workers)
                for val in unique_col:
                    encoding[col][val] = Index(unique_col).get_loc(val)
            data[col] = data[col].replace(encoding[col])

        return data, encoding
