import pandas as pd
import numpy as np
import geopy.distance

# Calculating distance between two localizations


def calcul_dist(data):
    print('Calculating distances between the airports localizations...')
    dist = np.ones(data.shape[0])
    for i in range(data.shape[0]):
        lat1 = data['lat_to'][i]
        lat2 = data['lat_from'][i]
        long1 = data['long_to'][i]
        long2 = data['long_from'][i]
        dist[i] = geopy.distance.distance((lat1, long1), (lat2, long2)).km
    Distance = pd.DataFrame(dist, columns=['distance'])
    return Distance


def getting_external_data(flights, airports, pop, US_holiday):
    # Add an id column to preserve the current order in prediction dataset
    flights["id"] = np.arange(flights.shape[0])
    # transforming date, into datetime format
    US_holiday['Date'] = pd.to_datetime(US_holiday['Date'])
    flights['date'] = pd.to_datetime(flights['flight_date'])
    flights.drop(columns={'flight_date'}, inplace=True)
    print("getting information about airports...")
    # Getting informations for Sourcee airports
    flights_extra = flights.merge(
        airports, left_on='from', right_on='local_code')
    flights_extra.rename(columns={"latitude_deg": "lat_from",
                                  "longitude_deg": "long_from",
                                  "elevation_ft": "elev_from",
                                  "municipality": "city_from",
                                  "score": "score_from"}, inplace=True)
    # Getting informations for Destination airports
    flights_extra = flights_extra.merge(
        airports, left_on='to', right_on='local_code')
    flights_extra.rename(columns={"latitude_deg": "lat_to",
                                  "longitude_deg": "long_to",
                                  "elevation_ft": "elev_to",
                                  "municipality": "city_to",
                                  "score": "score_to"}, inplace=True)

    flights_extra.drop(columns={'local_code_y', 'local_code_x'}, inplace=True)

    # Correcting city name
    flights_extra["city_from"].replace({"Newark": "New York"}, inplace=True)
    flights_extra["city_to"].replace({"Newark": "New York"}, inplace=True)

    print("Getting information about airports'cities...")
    # Getting cities pop info

    flights_extra = flights_extra.merge(
        pop, left_on='city_from', right_on='city')
    flights_extra.rename(
        columns={"population": "pop_from", "density": "dens_from"}, inplace=True)

    flights_extra = flights_extra.merge(
        pop, left_on='city_to', right_on='city')
    flights_extra.rename(
        columns={"population": "pop_to", "density": "dens_to"}, inplace=True)

    flights_extra.drop(columns={'city_x', 'city_y'}, inplace=True)

    # Calculating distance between
    Distance = calcul_dist(flights_extra)
    # Joining distance with the rest of the data

    flights_extra = flights_extra.join(Distance)

    # Drop coordinates

    flights_extra.drop(columns={'lat_from', 'long_from', 'long_to',
                       'lat_to', 'elev_to', 'elev_from', 'city_from', 'city_to'})

    # Scaling scores

    flights_extra['score_to'] = flights_extra['score_to']/1000000
    flights_extra['score_from'] = flights_extra['score_from']/1000000
    print("Sorting columns...")
    # Sorting columns
    if 'target' in flights_extra.columns:
        sorted_columns = ['id', 'date', 'from', 'score_from', "pop_from", "dens_from", 'to', 'score_to',
                          "pop_to", "dens_to", 'avg_weeks', 'std_weeks', 'distance', 'target']
    else:
        sorted_columns = ['id', 'date', 'from', 'score_from', "pop_from", "dens_from", 'to', 'score_to',
                          "pop_to", "dens_to", 'avg_weeks', 'std_weeks', 'distance']

    flights_extra = flights_extra.reindex(sorted_columns, axis=1)
    print("Extracting month and day from date...")
    # Extracting year and month from date
    flights_extra['month'] = [d.strftime('%b') for d in flights_extra.date]
    flights_extra['day'] = flights_extra['date'].dt.day
    print("Adding US holiday information...")
    flights_extra = flights_extra.merge(
        US_holiday, left_on='date', right_on='Date', how='left')
    flights_extra.drop(columns={"Date"}, inplace=True)

    flights_extra['Holiday'] = flights_extra['Holiday'].fillna(0)

    flights_extra.drop(columns=['date'], inplace=True)

    # One hot encoding and deleting original columns
    print('One hot encoding the airports and months...')
    source = pd.get_dummies(flights_extra['from'], prefix='from')
    destination = pd.get_dummies(flights_extra['to'], prefix='to')

    flights_extra = flights_extra.drop(columns={'from', 'to'})

    flights_extra = source.join(flights_extra)
    flights_extra = destination.join(flights_extra)

    months = pd.get_dummies(flights_extra['month'])
    flights_extra = months.join(flights_extra)
    flights_extra.drop(columns={'month'}, inplace=True)
    print("Sorting dataset according to the initial order...")
    # Sort values according to the initial order :
    flights_extra = flights_extra.sort_values(["id"], ascending=True)
    flights_extra.drop(columns={"id"}, inplace=True)
    flights_extra.reset_index(inplace=True, drop=True)
    print("THERE YOU GO !")
    return flights_extra


def completing_columns_and_order(data, to_predict):
    print('\n')
    print("Creating columns of months unexisting in validation set but existing in learning set and filling with 0...")
    # create the same columns for to_predict
    for i in list(set(data.columns) - set(to_predict.columns)):
        to_predict[i] = 0
    # drop the target name
    to_predict.drop(columns=['target'], axis=1, inplace=True)
    print("Rearranging columns the same way as the learning dataset...")
    # rearrange the columns accordin to learnin dataset
    to_predict = to_predict.reindex(
        data.drop(columns={'target'}).columns, axis=1)
    return to_predict
