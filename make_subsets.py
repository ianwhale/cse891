from save_subset import save_subset, name_to_id_dict


if __name__ == "__main__":
    for cat in ["MEUBLE", "ELECTRONIQUE"]:
        print("Making {} dictionary...".format(cat))
        d = name_to_id_dict(cat)

        print("Making {} subset...".format(cat))
        save_subset(d, cat, 'train.bson')