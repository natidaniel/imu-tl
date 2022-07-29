from os.path import join
import argparse
import pandas as pd
import pathlib
import os


def read_dataset(dataset_in_file):
    return pd.read_csv(dataset_in_file)


if __name__ == "__main__":
    # Parameters
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("inputPath", help="a path to IMU user data")
    arg_parser.add_argument("outputPath", help="a path save final data format (csv)")
    arg_parser.add_argument("task", help="name of task to perfom: 'Train', 'Test'")
    args = arg_parser.parse_args()

    raw_data = []
    for i, file in enumerate(os.listdir(args.inputPath)):
        if file.endswith(".csv"):
            if i == 0:  # first
                dfs = read_dataset(os.path.join(args.inputPath, file))
            else:
                df = read_dataset(os.path.join(args.inputPath, file))
                dfs = dfs.append(df,ignore_index = True)

    # WA for qualitest:
    # dfss = dfs[dfs.UserMode != 4]

    # WA for GPS:
    # remove the first raw
    dfs = dfs.iloc[1:, :]
    # remove unused classes
    dfs = dfs[dfs.label != 0]
    dfs = dfs[dfs.label != 5]
    dfs = dfs[dfs.label != 6]
    dfs = dfs[dfs.label != 7]
    dfs = dfs[dfs.label != 8]
    dfs = dfs[dfs.label != 9]
    dfs = dfs[dfs.label != 10]
    # aggregate to vehicle
    dfs = dfs.replace({'label': 11}, 4)
    dfs = dfs.replace({'label': 4}, 3)
    dfss = dfs.drop(['time', 'user','time_diff'], axis = 1)
    dfss.to_csv(join(args.outputPath,"GPS_" + args.task + "_set.csv"), index=False)
    print('End pre-process')