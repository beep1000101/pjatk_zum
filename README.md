

# TL;DR / Short Summary

Projekt zawiera trzy kompletne pipeline'y deep learningowe, każdy zrealizowany w formie notatnika Colab w katalogu `notebooks/`. Każdy pipeline odpowiada innemu zadaniu:

**Wszystkie notebooki treningowe korzystają z modułu `data_ingestion` do automatycznego pobierania i przygotowania danych wejściowych. Dzięki temu uruchomienie pipeline'u jest wygodne, powtarzalne i nie wymaga ręcznego pobierania plików.**

- Klasyfikacja sentymentu recenzji filmowych (IMDB)
- Rozpoznawanie komend głosowych (keyword spotting, mini_speech_commands)
- Multimodalne dopasowanie obraz–tekst (CLIP na CIFAR-10)

Wszystkie szczegóły implementacyjne, wyniki i artefakty znajdują się w odpowiednich notebookach.

---

# Opis modeli

W repozytorium znajdują się trzy modele/pipeline'y:

1. **Sentiment Embeddings**
    - Klasyfikacja sentymentu recenzji filmowych jako pozytywne/negatywne na podstawie zbioru IMDB.
2. **ASR Commands**
    - Rozpoznawanie krótkich komend głosowych (keyword spotting) na podstawie nagrań audio.
3. **CLIP Multimodal**
    - Dopasowanie obraz–tekst oraz klasyfikacja obrazów z CIFAR-10 przy użyciu modelu CLIP.

Szczegółowe opisy, założenia i wyniki:
- Znajdują się **wewnątrz notebooków** w katalogu `notebooks/`.
- Dodatkowe README dla każdego pipeline'u znajdują się w podkatalogach `notebooks/<pipeline>/`.

---

# Konfiguracja środowiska (opcjonalnie)

Projekt był rozwijany i testowany głównie w środowisku Google Colab.

**Lokalna instalacja nie jest rekomendowana** (większość użytkowników nie będzie jej potrzebować).

Jeśli jednak chcesz uruchomić środowisko lokalnie:


1. Zainstaluj [uv](https://github.com/astral-sh/uv) lub użyj pip:

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

lub

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Opcjonalnie) Skrypt `scripts/data_ingestion/clear_cache.sh` pozwala wyczyścić cache danych.

---

# Gdzie szukać szczegółowej dokumentacji?

- Szczegółowe opisy, kod i wyniki znajdują się **w notebookach** w katalogu `notebooks/`.
- Każdy pipeline posiada własny podkatalog z README oraz helperami.
- Dodatkowe pliki planistyczne i roadmapy: `.llm_planning/`.

---


# Kontakt i autor

Autor: Mateusz Poniatowski

Projekt końcowy na przedmiot "Zastosowania Uczenia Maszynowego" (PJATK).