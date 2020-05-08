import pandas as pd
import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path of the csv file.')
    parser.add_argument('--fr', default=30, help='Frame rate in frames per second',
                        type=int)
    parser.add_argument('--yi', default=1050, type=int)
    parser.add_argument('--yf', default=585, type=int)
    parser.add_argument('--dist', default=21.5)

    args = parser.parse_args()
    frame_rate = args.fr
    yi = args.yi
    yf = args.yf
    dist = args.dist
    # Bring in csv data
    df = pd.read_csv(args.file)

    # Make a new dataframe using groupby with index as object number, total
    # number of tracks, first and last y position and t.
    objects_df = df.groupby('o').agg(
        count=pd.NamedAgg(column='t', aggfunc='count'),
        y_0=pd.NamedAgg(column='y', aggfunc='first'),
        y_f=pd.NamedAgg(column='y', aggfunc='last'),
        t_0=pd.NamedAgg(column='t', aggfunc='first'),
        t_f=pd.NamedAgg(column='t', aggfunc='last'))

    #Drop objects for which there are less than 4 rows.
    objects_df.drop(objects_df[objects_df['count']<4].index, inplace=True)

    # Determine the direction traveled by each object
    objects_df['delta_y'] = objects_df['y_f'] - objects_df['y_0']
    objects_df['direction'] = objects_df['delta_y'].apply(lambda x: 'N' if x < 0 else 'S')
    objects_df.drop(['delta_y'], axis=1, inplace=True)

    # Determine the average number of cars per minute
    delta_t = (objects_df.t_f.max() - objects_df.t_0.min())/(60*frame_rate)
    cpm_N = objects_df['direction'].value_counts()['N']/delta_t
    cpm_S = objects_df['direction'].value_counts()['S']/delta_t
    print("Going north: {:.2f} cars per minute".format(cpm_N))
    print("Going south: {:.2f} cars per minute".format(cpm_S))

    s_avg = []
    for o in objects_df.index:
        if objects_df.loc[o, 'direction']=='S':
            s_avg.append(np.nan)
            continue
        df_mini = df[df['o']==o][['y', 't']]
        y1 = None
        t1 = None
        for _, row in df_mini.iterrows():
            y=row['y']
            t=row['t']
            if y1 and y < yi and y1 > yi:
                ti = t - (float((yi-y)*(t-t1))/(y1-y))
            if y1 and y < yf and y1 > yf:
                tf = t - (float((yf-y)*(t-t1))/(y1-y))
                break
            y1 = y
            t1 = t
        s_avg.append(2.237*dist*frame_rate/(tf - ti))
    objects_df['s_avg'] = s_avg
    print("Average speed for cars traveling north is: {:.2f} miles per hour.".format(objects_df.s_avg.mean()))

if __name__ == '__main__': main()
