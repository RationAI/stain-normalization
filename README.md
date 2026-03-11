# Stain Normalization

Tento repozitár slúži ako doplnkový materiál k bakalárskej práci **"Normalizácia farbenia histopatologických snímkov pomocou neuronových sietí"**.

Celý kód nie je možné spustiť samostatne, pretože vyžaduje prístup k citlivým dátam a k platforme na správu strojového učenia MLflow. Avšak je možné spustiť demo, ku ktorému je pripravených pár vzoriek na demonštráciu.

## Demo

Demo skript umožňuje načítať jeden obrázok alebo celý priečinok, normalizovať ich pomocou predtrénovaného modelu a uložiť výsledné obrázky do určeného priečinka.

## Priložené dáta

- **Originálne obrázky** (v priečinku `./demo_data/original`) – referenčné vzorky.  
- **Upravené obrázky** (v priečinku `./demo_data/modified`) – referenčné obrázky s modifikovaným sfarbením, ktoré sa následne modelom znormalizujú späť k originálnemu vzhľadu.  
- **Obrázky z iného datasetu** (v priečinku `./demo_data/to_predict`) – ukazujú, ako model dokáže prispôsobiť farby vstupov tak, aby zodpovedali farebnému štýlu trénovacích dát.  
- **Model checkpoint** 

## Spustenie dema

Projekt je spravovaný pomocou nástroja **pdm**, ktorý umožňuje jednoduchú správu závislostí a prostredia. Stačí nainštalovať závislosti pomocou:

```bash
pdm install
```
a potom spustiť demo skript:

```bash
pdm run python demo.py --input ./demo_data/modified
```

## Dostupné arguemnty:
- **input**: cesta k obrázku alebo priečinku s obrázkami na normalizáciu (default ./demo_data/modified)
- **output**: priečinok, kam sa uložia normalizované obrázky (default ./demo_data)
- **use_cpu**: defaultne nadstavené na použitie GPU ak je dostupná, avšak ak by nastali problémy odporúčam použivať iba CPU