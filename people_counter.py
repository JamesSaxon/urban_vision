#!/usr/bin/env python

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh', default = 0.4, help = "Confidence Threshold")
    parser.add_argument('--frame_rate', default = 30, help='Path of the labels file.')
    parser.add_argument('--file', help="Data File", required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.file, header = 0, names=['frame', 'x', 'y', 'xmin',
                                                   'ymin', 'xmax', 'ymax',
                                                   'area', 'label', 'conf'])

    df = df.astype({'frame': 'int32', 'x': float, 'y': float, 'xmin': float,
                    'ymin': float, 'xmax': float, 'ymax': float, 'area': float,
                    'conf': float})

    #Make a column for people with confidences above the threshold.
    df['detected'] = 0
    df.loc[df['conf']>=args.thresh, 'detected'] = 1

    count_df = pd.DataFrame(df.groupby('frame')['detected'].sum())
    count_df['rolling_avg'] = count_df.rolling(5, min_periods=1).mean()
    count_df = count_df.reset_index()
    count_df['t'] = count_df['frame']/(60*args.frame_rate)

    fig= plt.figure(figsize=(15,5))
    plt.plot(count_df['t'], count_df['rolling_avg'], '-')
    plt.xlabel('time (minutes)', fontsize=16)
    plt.ylabel('avg people count', fontsize=16)
    plt.show()

if __name__ == '__main__': main()
