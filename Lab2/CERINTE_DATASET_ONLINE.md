# — Set online: calitate vin roșu (UCI / Kaggle)


---

## Cerința 1 — Încărcare

Încărcați `winequality-red.csv` cu separator corect. Afișați `shape`, primele 5 rânduri, tipurile coloanelor.

## Cerința 2 — Valori lipsă

Implementați `nan_replace_df(tabel)` în `functii_online.py` (media pe coloană pentru numerice). Aplicați pe tabel; confirmați că nu rămân NaN (setul UCI nu are de obicei lipsuri — codul trebuie să fie **general**, ca la examen).

## Cerința 3 — Filtrare 

Salvați în `data_out/vinuri_calitate_min7.csv` toate vinurile cu **`quality >= 7`**. Păstrați toate coloanele.

## Cerința 4 — Sortare (ca „Prezenta_sort”)

Salvați în `data_out/top50_alcool.csv` cele **50** de vinuri cu cel mai mare **`alcohol`**, sortate descrescător.

## Cerința 5 — Agregare (ca „Regiuni.csv”)

Salvați în `data_out/medii_pe_calitate.csv` media coloanelor **`alcohol`** și **`pH`** pentru fiecare valoare a coloanei **`quality`** (un rând per notă de calitate).

## Cerința 6 — Coloană derivată (ca procente)

Adăugați coloana **`aciditate_totala`** = `fixed acidity` + `volatile acidity` + `citric acid`.  
Salvați în `data_out/vinuri_cu_aciditate.csv` doar coloanele:  
`quality`, `alcohol`, `aciditate_totala`, `pH`.

## Cerința 7 — Interpretare numerică

Determinați **nota de calitate** (`quality`) pentru care media **`alcohol`** este maximă.  
Salvați un fișier `data_out/calitate_max_alcool.csv` cu o singură coloană `quality` și o singură valoare.

## Cerința 8 — Corelații (pregătire ACP / preliminară)

Pentru coloanele:  
`fixed acidity`, `volatile acidity`, `citric acid`, `alcohol`, `quality`  
calculați matricea de corelație și salvați **`data_out/R.csv`** (cu nume de rând/coloană = numele variabilelor).

## Cerința 9 — Standardizare (pregătire ACP)

Creați **`alcohol_std`** = (alcohol − media) / deviația standard (populație sau eșantion — **documentați** în comentariu ce folosiți).  
Salvați `data_out/alcohol_standardizat.csv` cu coloanele `alcohol`, `alcohol_std`.

## Cerința 10 — Mini matrice de distanțe

Pe **primele 20 de vinuri** și pe coloanele standardizate  
`fixed acidity`, `volatile acidity`, `alcohol` (standardizați fiecare coloană),  
calculați matricea distanțelor **euclidiene** 20×20 (`scipy.spatial.distance.cdist` sau formula voastră).  
Salvați **`data_out/d_euclid_20.csv`**.

---

## Bonus

Funcția `salvare_matrice(x, index_linie, index_coloana, fisier)` în `functii_online.py` — refolosiți-o la cerința 10.

## Întrebări de interpretare (oral)

1. De ce corelația alcool–calitate nu implică cauzalitate?
2. De ce standardizăm înainte de distanțe euclidiene pe coloane cu unități diferite?
