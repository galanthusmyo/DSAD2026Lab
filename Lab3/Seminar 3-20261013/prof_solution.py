import pandas as pd

industria_alim = pd.read_csv("IndustriaAlimentara.csv", index_col=0)

print(industria_alim)

industrii = list(industria_alim.columns[1:])

print(industrii)

angajati = industria_alim[industrii].apply(
    func = lambda x : x.sum(),
    axis = 1
)

cerinta1 = industria_alim[angajati > 0]

cerinta1.to_csv("cerinta1.csv")

# task 3 - merge

population = pd.read_csv("PopulatieLocalitati.csv", index_col=0)

merged_df = industria_alim.merge(population, left_index=True, right_index=True)

print(merged_df)