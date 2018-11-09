import os
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



        # # Shuffle the dataset
        # self.df = self.df.sample(frac=1).reset_index(drop=True)
        #
        # # get the column of classes, and map the respective function
        # self.expected_outputs = self.df.iloc[:, [class_column]].values.tolist()
        # self.expected_outputs = list(map(function, self.expected_outputs))
        #
        # # Normalize dataset
        # self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        #
        # # Save dataframe as list
        # columns = [x for x in range(len(self.df.columns)) if x != class_column]
        # self.inputs = self.df.iloc[:, columns].values.tolist()



# # main
# dir = os.path.join(os.path.abspath(os.path.join(__file__ , "../../..")), os.path.join('datasets', 'seeds_dataset.csv'))
#
# dff = CSV_loader(dir, 7)
#
# print(dff.expected_outputs)
# print(dff.inputs)
