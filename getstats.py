import cdiscountdataset as cdd

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


def get_category_breakdown(dataframe):
    category_count = {}

    for i in range(len(dataframe.indexes)):
        category = dataframe.indexes.iloc[i][1]

        if category in category_count.keys():
            category_count[category] += 1
        else:
            category_count[category] = 1

    return category_count

def run_stats(dataframe):
    cat_counts = get_category_breakdown(dataframe)
    statsM = StatsManager(cat_counts)
    print(statsM)


if __name__ == "__main__":
    dataframe = cdd.CDiscountDataSet('train_example.bson')
    run_stats(dataframe)
