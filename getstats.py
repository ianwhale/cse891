import cdiscountdataset as cdd
from os.path import isfile
import pandas as pd


class StatsManager:

    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.categories_df = pd.read_csv('category_names.csv', index_col="category_id")
        self.categories_df["category_idx"] = pd.Series(range(len(self.categories_df)), index=self.categories_df.index)

        self.cat1_dict = self.get_level_breakdown(level=1)
        self.cat2_dict = self.get_level_breakdown(level=2)
        self.cat3_dict = self.get_level_breakdown(level=3)

    # def __str__(self):
    #     min_ = self.get_min()
    #     max_ = self.get_max()
    #     mean_ = self.get_mean()
    #     med_ = self.get_median()
    #
    #     s = "\nStats Manager:\n--------------\n" + \
    #         "MIN -> Index: {} Value: {}\n".format(min_, self.data[min_]) + \
    #         "MAX -> Index: {} Value: {}\n".format(max_, self.data[max_]) + \
    #         "MEAN -> {0:.2f}\n".format(mean_) + \
    #         "MEDIAN -> {0:.2f}\n".format(med_)
    #
    #     return s

    def csv_to_dict(self, file_name):
        csv_dict = {}
        file_ = open(file_name, 'r')
        data = file_.read().split('\n')
        if data[-1] == '': data.pop()
        file_.close()

        for line in data:
            key, value = line.split(',')
            csv_dict[key] = int(value)

        return csv_dict

    def write_dict(self, dict_to_write, file_name):
        print("Writing {} to file:".format(file_name))
        file_ = open(file_name, 'w')
        for key in dict_to_write.keys():
            file_.write(str(key) + "," + str(dict_to_write[key]) + '\n')

        print("Done")
        file_.close()

    # def get_min(self):
    #     min_key = min(self.data, key=self.data.get)
    #     return min_key
    #
    # def get_max(self):
    #     max_key = max(self.data, key=self.data.get)
    #     return max_key
    #
    # def get_mean(self):
    #     val_sum = 0
    #     for key in self.data: val_sum += self.data[key]
    #     return val_sum / len(self.data)
    #
    # def get_median(self):
    #     sort_data = sorted(self.data.values())
    #     len_data = len(sort_data)
    #     if len_data % 2 == 0:
    #         return (sort_data[len_data // 2 - 1] + sort_data[len_data // 2]) / 2
    #     else:
    #         return sort_data[len_data // 2]

    def get_category_breakdown(self, levelA, levelB):
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
    # statsM.run_stats()
