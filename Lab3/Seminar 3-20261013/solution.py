import pandas as pd
from pathlib import Path


FOLDER = Path(__file__).resolve().parent


ACTIVITIES = [
    "Carne", "Peste", "LegumeFructe", "Uleiuri",
    "Lactate", "Morarit", "Panificatie",
]

# Read the tables and attach population, county, and region to each locality.
food = pd.read_csv(FOLDER / "IndustriaAlimentara.csv")
population = pd.read_csv(FOLDER / "PopulatieLocalitati.csv")
counties = pd.read_csv(FOLDER / "Coduri_Judete.csv")
data = food.merge(population.drop(columns="Localitate"), on="Siruta")
data["Regiune"] = data["Judet"].map(counties.set_index("IndicativJudet")["Regiune"])

# 1. Keep localities with at least one food-industry employee.
data.loc[data[ACTIVITIES].sum(axis=1) > 0, food.columns].to_csv(
    FOLDER / "Cerinta1.csv", index=False
)

# 2. Each activity as a percentage of the locality's food-industry employees.
total = data[ACTIVITIES].sum(axis=1).replace(0, float("nan"))
percentages = data[ACTIVITIES].div(total, axis=0).fillna(0) * 100
result = data[["Siruta", "Localitate"]].join(percentages.round(2))
result.to_csv(FOLDER / "Cerinta2.csv", index=False)

# Sum population and employees for each county; tasks 3, 4, and 6 use this.
by_county = data.groupby("Judet")[ACTIVITIES + ["Populatie"]].sum()
county_population = by_county["Populatie"].replace(0, float("nan"))

# 3. Counties sorted by food-industry employees per 100 residents.
county_percent = by_county[ACTIVITIES].sum(axis=1).div(county_population) * 100
county_result = pd.DataFrame({
    "IndicativJudet": county_percent.index,
    "NumeJudet": county_percent.index.map(
        counties.set_index("IndicativJudet")["NumeJudet"]
    ),
    "PondereAngajatiInPopulatie(%)": county_percent.fillna(0).round(4).values,
}).sort_values("PondereAngajatiInPopulatie(%)", ascending=False)
county_result.to_csv(FOLDER / "Cerinta3.csv", index=False)

# 4. The activity with the most employees in each county.
dominant = by_county[ACTIVITIES].idxmax(axis=1)
pd.DataFrame({
    "IndicativJudet": dominant.index,
    "ActivitateDominanta": dominant.values,
}).to_csv(FOLDER / "Cerinta4.csv", index=False)

# 5. Population-weighted mean employees per locality in each region.
weighted_employees = data[ACTIVITIES].mul(data["Populatie"], axis=0)
weighted_by_region = weighted_employees.groupby(data["Regiune"]).sum()
region_population = data.groupby("Regiune")["Populatie"].sum().replace(0, float("nan"))
regional_average = weighted_by_region.div(region_population, axis=0).fillna(0).round(4)
regional_average.rename_axis("Regiune").reset_index().to_csv(
    FOLDER / "Cerinta5.csv", index=False
)

# 6. County activity rate divided by the same activity's national rate.
national_population = data["Populatie"].sum()
national_rate = data[ACTIVITIES].sum() / national_population
county_rate = by_county[ACTIVITIES].div(county_population, axis=0)
location_index = county_rate.div(national_rate.replace(0, float("nan"))).fillna(0)
location_index.round(4).rename_axis("IndicativJudet").reset_index().to_csv(
    FOLDER / "Cerinta6.csv", index=False
)
