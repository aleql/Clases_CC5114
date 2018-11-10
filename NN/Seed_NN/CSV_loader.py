
import pandas as pd

from Seed_NN.Dataset_input import Dataset_input
from auxiliar_methods.statistics_errors import identity


class CSV_loader:

    def __init__(self, csv_dir, class_column, function=identity):

        # Load csv as dataframe
        df = pd.read_csv(csv_dir)

        # Separate dataframe in train and test in 80/20 manner
        train = df.sample(frac=0.8, random_state=200)
        test = df.drop(train.index)

        self.train = Dataset_input(train, class_column, function)
        self.test = Dataset_input(test, class_column, function)
