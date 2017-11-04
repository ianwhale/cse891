Code to do image classification on the CDiscount Kaggle competition data.
See original challenge [here](https://www.kaggle.com/c/cdiscount-image-classification-challenge).

In order to get the `CDiscountDataSet` demo working on a fresh GCE instance do the following:

```
$ ./setup.sh
$ ./get_categories_data.sh
$ ./get_example_training.sh
$ python3 cdiscountdataset.py
```

The script `get_data.sh` should download the entire `training.bson` and `testing.bson` files from
the Kaggle challenge. So that script will likely take over 2 hours.