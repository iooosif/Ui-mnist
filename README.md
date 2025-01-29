Zadanie:
Vytvoril som doprednú neurónovú sieť (viacvrstvový perceptrón), ktorú som natrénoval pomocou troch optimalizačných algoritmov: SGD, SGD s momentom a Adam. Počas tréningu som sledoval trénovaciu a testovaciu chybu, ako aj presnosť modelu na testovacej množine (koľkokrát model správne predpovedal triedu). Cieľom bolo dosiahnuť presnosť vyššiu ako 97 %. Na implementáciu som použil knižnicu PyTorch.
Postup:
1.	Načítanie datasetu: Dataset MNIST som načítal priamo z PyTorch knižnice, kde je voľne dostupný. Dáta som rozdelil na trénovaciu a testovaciu množinu.
2.	Predspracovanie dát: Hodnoty pixelov som normalizoval do intervalu od -1 do 1. Cieľové hodnoty (čísla od 0 do 9) som použil priamo, bez potreby one-hot kódovania, pretože som využíval CrossEntropyLoss, ktorá ho nahrádza.
3.	Finalne navrhovanie modelu a hyperparametrov:
o	Model má tri vrstvy:
	Prvá skrytá vrstva obsahuje 256 neurónov a používa aktivačnú funkciu ReLU.
	Druhá vrstva má 128 neurónov, tiež s ReLU.
	Výstupná vrstva má 10 neurónov (pre každú triedu).
o	Hyperparametre:
	Rýchlosť učenia: 0,01 pre SGD a SGD s momentom, 0,0005 pre Adam.
	 Batch: 256.
	Počet epoch: 10 pre všetky modely.
4.	Tréning modelu: Každý model som natrénoval samostatne rovnaký počet epoch, aby som mohol porovnať ich výkon. Výsledky som sledoval vo forme zmien chyby a presnosti na trénovacej aj testovacej množine počas tréningu.
5.	Porovnanie optimalizačných algoritmov: Na základe trénovacej a testovacej presnosti som určil, ktorý optimalizačný algoritmus je najlepší.
![image](https://github.com/user-attachments/assets/a1d399c4-90c9-472a-8719-1f3293d602bb)
