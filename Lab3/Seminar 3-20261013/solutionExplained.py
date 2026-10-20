"""Pandas version of Tema 2-4, with explanations for each step."""

import pandas as pd
from pathlib import Path


FOLDER = Path(__file__).resolve().parent


ACTIVITIES = [
    "Carne", "Peste", "LegumeFructe", "Uleiuri",
    "Lactate", "Morarit", "Panificatie",
]

# A DataFrame is pandas' table. read_csv turns each CSV into a DataFrame.
# merge matches rows with the same Siruta, like a table join. The population
# file's Localitate column is dropped because the food file already has it.
# map uses the county code to look up the county's region.
food = pd.read_csv(FOLDER / "IndustriaAlimentara.csv")
population = pd.read_csv(FOLDER / "PopulatieLocalitati.csv")
counties = pd.read_csv(FOLDER / "Coduri_Judete.csv")
data = food.merge(population.drop(columns="Localitate"), on="Siruta")
data["Regiune"] = data["Judet"].map(counties.set_index("IndicativJudet")["Regiune"])

# 1. sum(axis=1) adds the activity columns across each row (one locality).
# Keep rows whose total is greater than zero. food.columns selects the same
# original columns required by the assignment.
data.loc[data[ACTIVITIES].sum(axis=1) > 0, food.columns].to_csv(
    FOLDER / "Cerinta1.csv", index=False
)

# 2. First find each locality's total employees. Replace a zero total with NaN
# so division is safe; fillna(0) then changes those undefined percentages to 0.
# div(..., axis=0) divides each row by that locality's total. Multiply by 100
# to express the result as percentages.
total = data[ACTIVITIES].sum(axis=1).replace(0, float("nan"))
percentages = data[ACTIVITIES].div(total, axis=0).fillna(0) * 100
result = data[["Siruta", "Localitate"]].join(percentages.round(2))
result.to_csv(FOLDER / "Cerinta2.csv", index=False)

# groupby("Judet") puts localities into county groups; sum adds each group's
# activity counts and population. These county totals are reused below.
by_county = data.groupby("Judet")[ACTIVITIES + ["Populatie"]].sum()
county_population = by_county["Populatie"].replace(0, float("nan"))

# 3. Add the seven activity totals, divide by county population, and multiply
# by 100. Build a result table, add county names with map, and sort descending.
county_percent = by_county[ACTIVITIES].sum(axis=1).div(county_population) * 100
county_result = pd.DataFrame({
    "IndicativJudet": county_percent.index,
    "NumeJudet": county_percent.index.map(
        counties.set_index("IndicativJudet")["NumeJudet"]
    ),
    "PondereAngajatiInPopulatie(%)": county_percent.fillna(0).round(4).values,
}).sort_values("PondereAngajatiInPopulatie(%)", ascending=False)
county_result.to_csv(FOLDER / "Cerinta3.csv", index=False)

# 4. idxmax finds the column name with the largest value in each county row.
dominant = by_county[ACTIVITIES].idxmax(axis=1)
pd.DataFrame({
    "IndicativJudet": dominant.index,
    "ActivitateDominanta": dominant.values,
}).to_csv(FOLDER / "Cerinta4.csv", index=False)

# 5. A weighted mean is sum(value * weight) / sum(weight). Here each locality's
# employee count is the value and its population is the weight. groupby sums
# the products and population within each region, then div calculates the mean.
weighted_employees = data[ACTIVITIES].mul(data["Populatie"], axis=0)
weighted_by_region = weighted_employees.groupby(data["Regiune"]).sum()
region_population = data.groupby("Regiune")["Populatie"].sum().replace(0, float("nan"))
regional_average = weighted_by_region.div(region_population, axis=0).fillna(0).round(4)
regional_average.rename_axis("Regiune").reset_index().to_csv(
    FOLDER / "Cerinta5.csv", index=False
)

# 6. The assignment's index is (county activity / national activity) divided
# by (county population / national population). This equals the county's
# activity employees per resident divided by the national rate. Above 1 means
# the activity is more concentrated in that county than nationally.
national_population = data["Populatie"].sum()
national_rate = data[ACTIVITIES].sum() / national_population
county_rate = by_county[ACTIVITIES].div(county_population, axis=0)
location_index = county_rate.div(national_rate.replace(0, float("nan"))).fillna(0)
location_index.round(4).rename_axis("IndicativJudet").reset_index().to_csv(
    FOLDER / "Cerinta6.csv", index=False
)
