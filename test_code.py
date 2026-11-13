import pandas as pd
import numpy as np
from my_tool import get_cwf
from my_tool import join_path
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utilsforecast.processing import make_future_dataframe

cwf = get_cwf()
fpath = join_path(cwf, r'data/vic_elec.csv')
vic_elec = (
    pd.read_csv(fpath, parse_dates=['ds'])
    .loc[lambda x: x["ds"] >= "2014"]
    .pivot(index=["ds", "Holiday"], columns="unique_id", values="y")
    .reset_index(level="Holiday").rename_axis(None, axis="columns")
)
vic_elec_daily = (
    vic_elec
    .resample("D")
    .agg({"Demand": "sum", "Temperature": "max", "Holiday": "first"})
    .reset_index()
    .assign(
        unique_id="VIC",
        Demand=lambda x: x["Demand"] / 1e3,
        Weekend=lambda x: x["ds"].dt.dayofweek.isin([5, 6]),
        Day_Type=lambda x: np.select(
            [x["Holiday"], ~x["Holiday"] & x["Weekend"]],
            ["Holiday", "Weekend"],
            default="Weekday",
        )
    )
)
ex_columns = ["unique_id", "ds", "Demand", "Temperature"]
vic_elec_daily_ex = vic_elec_daily[ex_columns].assign(**{
    "Weekday": vic_elec_daily["Day_Type"] == "Weekday",
    "Temperature**2": vic_elec_daily["Temperature"]**2,
})

sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D")
sf = sf.fit(df=vic_elec_daily_ex, target_col="Demand")

# print(vic_elec_daily_ex)

uids = vic_elec_daily_ex["unique_id"].unique()
last_times = vic_elec_daily_ex[["ds"]].max()
future_predictors = (
    make_future_dataframe(uids, last_times, freq="D", h=14)
    .assign(**{
        "Weekday": lambda x: ~x["ds"].dt.dayofweek.isin([5, 6]),
        "Temperature": 26,
        "Temperature**2": 26**2,
    })
)

print(future_predictors)

level = [80, 95]
# Ensure column order agrees with usage above in sf.fit()
columns = vic_elec_daily_ex.columns.drop('Demand')
fc = sf.predict(h=14, X_df=future_predictors[columns], level=level)