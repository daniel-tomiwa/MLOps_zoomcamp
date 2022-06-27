import os
import pickle
import pandas as pd
import argparse


model_path = "model.bin"
taxi_type = "fhv"


def read_data(filename: str):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    dicts = df[categorical].to_dict(orient='records')
    
    return dicts

def load_model(model_path: str):
    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr

def apply_model(input_file: str, output_file: str, year: int, month: int):

    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    dicts = prepare_dictionaries(df)
    dv, lr = load_model(model_path)

    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df_result = pd.DataFrame()
    df_result["ride_id"] = df.ride_id.copy()
    df_result["predictions"] = y_pred
    
    # df_result.to_parquet(
    #     output_file,
    #     engine='pyarrow',
    #     compression=None,
    #     index=False
    # )

    return df_result

def get_paths(year, month):

    input_file = f"https://nyc-tlc.s3.amazonaws.com/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_path = f"output/{taxi_type}"

    return input_file, output_path

def run(argparser):
    year = int(argparser.year)
    month = int(argparser.month)

    input_file, output_path = get_paths(year, month)
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)
    # output_file = os.path.join(output_path, f"{year:04d}-{month:02d}.parquet")

    result = apply_model(
             input_file=input_file,
             output_file=output_path,
             year=year,
             month=month
             )

    print(f"The mean prediction for the year:{year}, and month:{month} for {taxi_type} is:", result.predictions.mean())

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="The desired month and year result to be scored with the model")

    parser.add_argument(
        "--year",
        default=2021,
        help="the year of the taxi trip data to be scored."
    )

    parser.add_argument(
        "--month",
        default=2,
        help="the month of the taxi trip data to be scored."
    )
    args = parser.parse_args()

    run(args)