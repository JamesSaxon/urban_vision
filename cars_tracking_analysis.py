import pandas as pd
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path of the csv file.')
    parser.add_argument('--fr', default=30, help='Frame rate in frames per second',
                        type=int)

    args = parser.parse_args()
    frame_rate = args.fr

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

if __name__ == '__main__': main()
