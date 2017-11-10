import cdiscountdataset as cdd
from os.path import isfile
import pandas as pd


class StatsManager:

    def __init__(self, dataframe):
        """
        Constructor.
        Creates files needed for basic stats.

        :param dataframe: CDiscountDataset to create stats manager
        """
        self.dataframe = dataframe

        self.categories_df = pd.read_csv('category_names.csv', index_col="category_id")
        self.categories_df["category_idx"] = pd.Series(range(len(self.categories_df)), index=self.categories_df.index)

        self.cat1_dict = self.get_level_breakdown(level=1)
        self.cat2_dict = self.get_level_breakdown(level=2)
        self.cat3_dict = self.get_level_breakdown(level=3)

    def csv_to_dict(self, file_name):
        """
        Function to read given csv file into a dict

        :param file_name: Name of csv file
        :return: Dict of csv data
        """
        csv_dict = {}

        file_ = open(file_name, 'r')
        data = file_.read().split('\n')
        file_.close()

        if data[-1] == '':
            data.pop()

        for line in data:
            key, value = line.split(',')
            csv_dict[key] = int(value)

        return csv_dict

    def write_dict(self, dict_to_write, file_name):
        """
        Writes stats dict to file.

        :param dict_to_write:
        :param file_name:
        :return:
        """
        print("Writing {} to file:".format(file_name))
        file_ = open(file_name, 'w')
        for key in dict_to_write.keys():
            file_.write(str(key) + "," + str(dict_to_write[key]) + '\n')

        print("Done")
        file_.close()

    def get_category_breakdown(self, levelA, levelB):
        """
        Count number of unique categories for levelB given a levelA category

        :param levelA: integer to denote superset level
        :param levelB: integer to denote subset level
        :return: dict with levelA categories as key, count of unique categories as value
        """
        level_a_keys = self.categories_df["category_level{}".format(levelA)].unique()
        category_count = {k: 0 for k in level_a_keys}

        prev_b = ''
        for k in range(len(self.categories_df)):
            category_a = self.categories_df["category_level{}".format(levelA)].iloc[k]
            category_b = self.categories_df["category_level{}".format(levelB)].iloc[k]
            if category_b == prev_b:
                continue

            prev_b = category_b
            category_count[category_a] += 1

        return category_count

    def get_level_breakdown(self, level):
        """
        Get number of images in data set for each category in a level

        :param level: integer to denote what level to get breakdown from
        :return:
        """
        file_name = 'category_count_level{}.csv'.format(level)

        if isfile(file_name):
            print("Found", file_name, "Loading...")
            return(self.csv_to_dict(file_name))

        print("Generating {}...".format(file_name))

        level_path = "category_level{}".format(level)
        unique_list = self.categories_df[level_path].unique()
        category_count = {k: 0 for k in unique_list}
        data_len = len(self.dataframe.indexes)

        for i in range(data_len):
            if i % (data_len // 10) == 0: print('\r{0:.1f}%'.format((i / data_len) * 100), end='')

            category_idx = self.dataframe.indexes.iloc[i][1]
            cat_name = self.categories_df[level_path].iloc[category_idx]
            category_count[cat_name] += 1

        self.write_dict(category_count, file_name)
        return category_count

    def run_stats(self):
        pass


if __name__ == "__main__":
    dataframe = cdd.CDiscountDataSet('train.bson')
    statsM = StatsManager(dataframe)
    statsM.write_dict(statsM.get_category_breakdown(1, 2), "level_1-2_cat_breakdown.csv")
    statsM.write_dict(statsM.get_category_breakdown(1, 3), "level_1-3_cat_breakdown.csv")
    statsM.write_dict(statsM.get_category_breakdown(2, 3), "level_2-3_cat_breakdown.csv")
