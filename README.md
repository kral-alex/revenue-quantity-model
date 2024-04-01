# revenue-quantity-model

## Instalace
Základním předpokladem instalace je přítomnost balíčku Python 3.
Nejdříve je nutné získat repozitář. Pokud máte nainstalovaný balíček git, nejjednodušší je naklonování z GitHubu:

$ git clone https://github.com/kral-alex/revenue-quantity-model.git

Poté nainstalujte dependence:

$ pip install requirements.txt

Následně je program připraven na spuštění.

## Spuštění

Program se spouští pomocí příkazu:

$ python3 main.py cesta_k_datům počet_k_zpracování

kde cesta_k_datům je cesta ke složce se soubory 0_P.csv a 0_Q.csv, které obsahují ve formátu csv matici cen a matici kvantit s řádky v časové posloupnosti a sloupci jednotlivých produktů. Je možné poskytnout i soubory 0_H.csv a 0_I.csv obsahující názvy produtků a datetime záznamu jako řádky.

Výstupem z programu jsou řádky python slovníků pro každý interval změny ceny s klíči: 
1) price a time, které obsahují slovník vypočtených koeficientů pro každou implementaci obou modelů, 
2) klíčem PED, který obsahuje vypočtené PED pro každou implementaci obou modelů,
3) klíčem revenue, vypočtený průměr výnosu za druhou polovinu intervalu kolem změny ceny,
3) a klíčem datetime, datum změny ceny v intervalu.
