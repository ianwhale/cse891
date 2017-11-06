import cdiscountdataset as cdd
import pandas as pd

class StatsManager:

    def __init__(self, data):
        self.data = data

    def __str__(self):
        min_ = self.get_min()
        max_ = self.get_max()
        mean_ = self.get_mean()
        med_ = self.get_median()

        s = "\nStats Manager:\n--------------\n" + \
            "MIN -> Index: {} Value: {}\n".format(min_, self.data[min_]) + \
            "MAX -> Index: {} Value: {}\n".format(max_, self.data[max_]) + \
            "MEAN -> {0:.2f}\n".format(mean_) + \
            "MEDIAN -> {0:.2f}\n".format(med_)

        return s

    def get_min(self):
        min_key = min(self.data, key=self.data.get)
        return min_key

    def get_max(self):
        max_key = max(self.data, key=self.data.get)
        return max_key

    def get_mean(self):
        val_sum = 0
        for key in self.data: val_sum += self.data[key]
        return val_sum / len(self.data)


    def get_median(self):
        sort_data = sorted(self.data.values())
        len_data = len(sort_data)
        if len_data % 2 == 0:
            return (sort_data[len_data // 2 - 1] + sort_data[len_data // 2]) / 2
        else:
            return sort_data[len_data // 2]


def get_level_breakdown(dataframe, unique_list, categories_df, level):
    category_count = {k: 0 for k in unique_list}

    for i in range(len(dataframe.indexes)):
        if i % 10000 == 0:
            print(i)
        category_idx = dataframe.indexes.iloc[i][1]
        level_path = "category_level{}".format(level)
        cat_name = categories_df[level_path].iloc[category_idx]
        category_count[cat_name] += 1


    return category_count

def run_stats(dataframe):
    categories_df = pd.read_csv('category_names.csv', index_col="category_id")
    categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)
    unique_list = categories_df["category_level3"].unique()

    print("Start Category Counting:", len(unique_list))
    cat_counts = get_level_breakdown(dataframe, unique_list, categories_df, 3)
    print("Writing to file:")
    file_ = open('stats3.csv', 'w')
    for key in cat_counts.keys():
        file_.write(str(key) + "," + str(cat_counts[key]) + '\n')

    print("Done")
    file_.close()


    # cat_counts = get_level_breakdown(dataframe, categories_df, 1)
    # exit(0)
    # statsM = StatsManager(cat_counts)
    # print(statsM)


if __name__ == "__main__":
    dataframe = cdd.CDiscountDataSet('train_example.bson')
    run_stats(dataframe)
