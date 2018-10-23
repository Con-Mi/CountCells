from os import listdir
import pandas as pd

def get_ids_in_csv():
    list_ids_train_input = [f for f in listdir("../data/GenData/images/")]
    list_ids_train_labels = [f for f in listdir("../data/GenData/labels/")]

    list_ids_train_input = sorted(list_ids_train_input)
    list_ids_train_labels = sorted(list_ids_train_labels)

    dict_train_input = {"ids": list_ids_train_input}
    dict_train_labels = {"ids": list_ids_train_labels}

    del list_ids_train_input, list_ids_train_labels
    train_input_df = pd.DataFrame(dict_train_input, index=None)
    train_labels_df = pd.DataFrame(dict_train_labels, index=None)
    train_input_df.to_csv("../data/GenData/train_input_ids.csv")
    train_labels_df.to_csv("../data/GenData/train_labels_ids.csv")

get_ids_in_csv()