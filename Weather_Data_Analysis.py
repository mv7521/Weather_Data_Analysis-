
import pandas as pd
import numpy as np
import json
from datetime import date, timedelta
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import URLError

cities = {
    "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319),
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "jaipur": (26.9124, 75.7873),
    "kolkata": (22.5726, 88.3639),
    "chennai": (13.0827, 80.2707),
    "bengaluru": (12.9716, 77.5946)
}


def get_weather_data(city):
    latitude, longitude = cities[city]

    end_date = date.today() - timedelta(days=7)
    start_date = end_date - timedelta(days=5 * 365)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": "auto"
    }

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        + urlencode(params)
    )

    print("\nDownloading historical weather data...")

    with urlopen(url, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    dates = data["daily"]["time"]
    temperatures = data["daily"]["temperature_2m_mean"]

    df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Temperature": temperatures
    })

    df["Temperature"] = pd.to_numeric(
        df["Temperature"], errors="coerce"
    )

    df = df.dropna().sort_values("Date").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("Not enough historical weather data received.")

    return df


def create_features(dates, start_date):
    dates = pd.to_datetime(dates)

    days = (dates - start_date).days
    years = np.asarray(days, dtype=float) / 365.25

    day_of_year = dates.dayofyear.to_numpy()
    angle = 2 * np.pi * day_of_year / 365.25

    features = np.column_stack([
        np.ones(len(dates)),
        years,
        np.sin(angle),
        np.cos(angle),
        np.sin(2 * angle),
        np.cos(2 * angle)
    ])

    return features


def train_model(df):
    split = int(len(df) * 0.80)

    train = df.iloc[:split]
    test = df.iloc[split:]

    start_date = df["Date"].min()

    x_train = create_features(train["Date"], start_date)
    y_train = train["Temperature"].to_numpy()

    x_test = create_features(test["Date"], start_date)
    y_test = test["Temperature"].to_numpy()

    model = np.linalg.lstsq(
        x_train, y_train, rcond=None
    )[0]

    predictions = x_test @ model

    mae = np.mean(np.abs(y_test - predictions))

    print("\nMODEL PERFORMANCE")
    print("------------------------------")
    print(f"Training records: {len(train)}")
    print(f"Testing records: {len(test)}")
    print(f"Mean Absolute Error: {mae:.2f} °C")

    return start_date


def predict_future(df, start_date):
    x_all = create_features(df["Date"], start_date)
    y_all = df["Temperature"].to_numpy()

    model = np.linalg.lstsq(
        x_all, y_all, rcond=None
    )[0]

    future_dates = pd.date_range(
        start=df["Date"].max() + timedelta(days=1),
        periods=30,
        freq="D"
    )

    x_future = create_features(future_dates, start_date)
    future_temperatures = x_future @ model

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Temperature": np.round(
            future_temperatures, 2
        )
    })

    return future_df


def main():
    print("======================================")
    print("   WEATHER DATA ANALYSIS AND PREDICTION")
    print("======================================")

    print("\nAvailable Cities:")
    print(", ".join(city.title() for city in cities))

    city = input("\nEnter city name: ").strip().lower()

    if city not in cities:
        print("City not available in the list.")
        return

    try:
        df = get_weather_data(city)

        print("\nHISTORICAL WEATHER ANALYSIS")
        print("--------------------------------------")
        print("City:", city.title())
        print("Total records:", len(df))
        print("Start date:", df["Date"].min().date())
        print("End date:", df["Date"].max().date())

        print("\nTemperature Statistics:")
        print(df["Temperature"].describe().round(2))

        monthly = (
            df.set_index("Date")["Temperature"]
            .resample("ME")
            .mean()
        )

        print("\nLast 12 Monthly Average Temperatures:")
        print(monthly.tail(12).round(2))

        yearly = (
            df.set_index("Date")["Temperature"]
            .resample("YE")
            .mean()
        )

        print("\nYearly Average Temperatures:")
        print(yearly.round(2))

        df.to_csv("weather_historical.csv", index=False)

        start_date = train_model(df)

        future_df = predict_future(df, start_date)

        future_df.to_csv("weather_predictions.csv", index=False)

        print("\nNEXT 30 DAYS TEMPERATURE TREND")
        print("--------------------------------------")
        print(future_df.to_string(index=False))

        print("\n======================================")
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("======================================")
        print("Historical data saved: weather_historical.csv")
        print("Predictions saved: weather_predictions.csv")

        print("\nNote: These are statistical trend estimates,")
        print("not official daily weather forecasts.")

    except URLError as error:
        print("\nInternet connection problem:", error)

    except Exception as error:
        print("\nAn error occurred:", error)


if __name__ == "__main__":
    main()
