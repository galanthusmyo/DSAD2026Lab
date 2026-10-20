"""Pandas solution with detailed, line-by-line explanations."""

# Import pandas, a library for working with tables. "pd" is its conventional
# short name, so later we write pd.read_csv instead of pandas.read_csv.
import pandas as pd
from pathlib import Path

# Path of this script's folder. Using it means input/output files are found
# beside the script even if you launch Python from a different folder.
FOLDER = Path(__file__).resolve().parent

# This script defines no custom functions or subfunctions. It calls library
# methods and Python built-ins. Main methods used below:
# read_csv(file) reads a CSV into a DataFrame (pandas table).
# DataFrame({...}) creates a table from named columns and their values.
# merge(other, on=...) joins tables by matching key values.
# drop(columns=...) removes column(s); set_index(col) uses a column for lookup.
# map(lookup) replaces each value with its matching value from a lookup table.
# groupby(key) groups table rows by a shared key, like county or region.
# sum(axis=1) adds across columns in each row; axis=1 means row-wise.
# div(x, axis=0) / mul(x, axis=0) divide/multiply each row by matching x value.
# replace(a, b) replaces values; fillna(x) replaces missing values with x.
# join(table) attaches columns by row labels (index); round(n) rounds decimals.
# idxmax(axis=1) returns the column name of the largest value in each row.
# sort_values(col) sorts rows; rename_axis names row labels; reset_index turns
#   row labels into a regular column. to_csv(file, index=False) saves a CSV.
# Python built-ins: float(...) converts to a decimal number, list(...) makes a
# list, and sum(...) adds values. Brackets select a column/value or list items.


# Save the seven activity column names in one list so the code can select or
# calculate across them all without repeating the same names everywhere.
ACTIVITIES = [
    "Carne", "Peste", "LegumeFructe", "Uleiuri",
    "Lactate", "Morarit", "Panificatie",
]

# Read each input CSV. read_csv returns a pandas DataFrame (a labeled table).
food = pd.read_csv(FOLDER / "IndustriaAlimentara.csv")
population = pd.read_csv(FOLDER / "PopulatieLocalitati.csv")
counties = pd.read_csv(FOLDER / "Coduri_Judete.csv")

# The population table also has Localitate, so drop that duplicate column.
# merge joins the two tables using matching Siruta codes. The default is an
# inner join: only codes found in both tables remain in data.
data = food.merge(population.drop(columns="Localitate"), on="Siruta")

# Use county code as a lookup key and select its region column. map looks up
# each locality's county code and adds the matching region to data.
data["Regiune"] = data["Judet"].map(counties.set_index("IndicativJudet")["Regiune"])

# TASK 1: Keep only localities with one or more food-industry employees.
# data[ACTIVITIES] selects the activity columns. sum(axis=1) adds across columns
# for every row (locality). > 0 creates a True/False filter for those rows.
# loc[row_filter, columns] selects rows and columns. food.columns asks for the
# original input columns. to_csv writes the result; index=False omits row numbers.
data.loc[data[ACTIVITIES].sum(axis=1) > 0, food.columns].to_csv(
    FOLDER / "Cerinta1.csv", index=False
)

# TASK 2: Calculate the share of workers in each activity at each locality.
# sum(axis=1) makes one total per locality. Replace zero with NaN (not-a-number)
# to avoid division by zero. float("nan") is Python's floating-point NaN value.
total = data[ACTIVITIES].sum(axis=1).replace(0, float("nan"))

# div(total, axis=0) divides each activity value by the total in its row.
# fillna(0) changes undefined results for zero-worker localities into zero.
# Multiplying by 100 changes fractions into percentages.
percentages = data[ACTIVITIES].div(total, axis=0).fillna(0) * 100

# Select locality code/name, then join on row labels to attach percentage
# columns. round(2) rounds each percentage to two decimal places.
result = data[["Siruta", "Localitate"]].join(percentages.round(2))
# Save the table to Cerinta2.csv, without adding pandas row numbers.
result.to_csv(FOLDER / "Cerinta2.csv", index=False)

# Prepare county totals for tasks 3, 4, and 6. groupby("Judet") groups localities
# with the same county code. ACTIVITIES + ["Populatie"] makes a list containing
# those seven columns plus population. sum() adds them within each county group.
by_county = data.groupby("Judet")[ACTIVITIES + ["Populatie"]].sum()

# Select the population total for each county. Replace zero with NaN to make
# future divisions safe. This is a Series: one labeled value per county.
county_population = by_county["Populatie"].replace(0, float("nan"))

# TASK 3: Add the county's seven activity totals, divide by population, and
# multiply by 100. .div aligns values by county, so each county uses its own
# population. The result is the number of food-industry workers per 100 people.
county_percent = by_county[ACTIVITIES].sum(axis=1).div(county_population) * 100

# Construct the output table from named columns. county_percent.index contains
# the county codes because groupby uses the group names as row labels (index).
county_result = pd.DataFrame({
    "IndicativJudet": county_percent.index,
    # Make codes lookup keys, select county names, and map each code to a name.
    "NumeJudet": county_percent.index.map(
        counties.set_index("IndicativJudet")["NumeJudet"]
    ),
    # Replace missing percentages with 0, round to 4 decimals, and take values
    # without their old Series index to make a plain DataFrame column.
    "PondereAngajatiInPopulatie(%)": county_percent.fillna(0).round(4).values,
}).sort_values("PondereAngajatiInPopulatie(%)", ascending=False)
# Sort descending (largest percentage first) and save the county ranking.
county_result.to_csv(FOLDER / "Cerinta3.csv", index=False)

# TASK 4: Find the largest activity count in each county. idxmax(axis=1) checks
# every county row and returns the name of its largest activity column. If tied,
# the first tied activity in ACTIVITIES is returned.
dominant = by_county[ACTIVITIES].idxmax(axis=1)
# Build a two-column table: dominant.index has county codes and dominant.values
# has activity names. Save the table without the extra pandas row-number column.
pd.DataFrame({
    "IndicativJudet": dominant.index,
    "ActivitateDominanta": dominant.values,
}).to_csv(FOLDER / "Cerinta4.csv", index=False)

# TASK 5: Weighted average formula is sum(value * weight) / sum(weight).
# Here value is locality employees and weight is locality population. mul(...,
# axis=0) multiplies all activity counts in each row by that row's population.
weighted_employees = data[ACTIVITIES].mul(data["Populatie"], axis=0)
# Group products by region and add them; these are weighted-average numerators.
weighted_by_region = weighted_employees.groupby(data["Regiune"]).sum()
# Add populations by region to get the weighted-average denominator. Replacing
# zero with NaN prevents division by zero on the next line.
region_population = data.groupby("Regiune")["Populatie"].sum().replace(0, float("nan"))
# Divide each region's weighted sums by its population; replace missing answers
# with zero and round to four decimal places.
regional_average = weighted_by_region.div(region_population, axis=0).fillna(0).round(4)
# Name the row labels, make them a normal Regiune column, and save the CSV.
regional_average.rename_axis("Regiune").reset_index().to_csv(
    FOLDER / "Cerinta5.csv", index=False
)

# TASK 6: The assignment's index is (county activity / national activity)
# divided by (county population / national population). This equals the county
# activity employees-per-resident rate divided by the national rate. Above 1
# means relatively more concentrated in the county; below 1 means less.
# First calculate national population and employee-per-resident rates.
national_population = data["Populatie"].sum()
national_rate = data[ACTIVITIES].sum() / national_population
# Calculate county activity rates by dividing employee totals by population.
county_rate = by_county[ACTIVITIES].div(county_population, axis=0)
# Divide each county rate by its matching national rate. Replace zero national
# rates with NaN before dividing; fillna(0) gives a safe zero in the output.
location_index = county_rate.div(national_rate.replace(0, float("nan"))).fillna(0)
# Round to four decimal places, name the row labels, turn them into a column,
# and save Cerinta6.csv without pandas' default row numbers.
location_index.round(4).rename_axis("IndicativJudet").reset_index().to_csv(
    FOLDER / "Cerinta6.csv", index=False
)
