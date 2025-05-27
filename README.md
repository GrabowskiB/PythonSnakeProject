# Snake AI z Wykorzystaniem Deep Q-Network (DQN)

![Snake Game Screenshot/GIF](Snake.gif) 
*Przykład wizualizacji działania agenta w grze.*

Projekt implementacji klasycznej gry Snake oraz agenta opartego na sztucznej inteligencji, który uczy się grać autonomicznie przy użyciu algorytmu Deep Q-Network (DQN).

**Link do Repozytorium:** [https://github.com/GrabowskiB/PythonSnakeProject.git](https://github.com/GrabowskiB/PythonSnakeProject.git)

## Spis Treści
- [Wprowadzenie](#wprowadzenie)
- [Cel Projektu](#cel-projektu)
- [Technologie](#technologie)
- [Struktura Projektu](#struktura-projektu)
- [Instalacja i Uruchomienie](#instalacja-i-uruchomienie)
- [Agent DQN](#agent-dqn)
- [Wyniki Treningu](#wyniki-treningu)
- [Możliwe Ulepszenia](#możliwe-ulepszenia)
- [Autor](#autor)

## Wprowadzenie
Projekt ten prezentuje implementację gry Snake oraz agenta AI wykorzystującego Deep Q-Network (DQN) do nauki optymalnej strategii gry. Środowisko gry stworzono w Pygame, a sieć neuronową agenta w TensorFlow/Keras.

## Cel Projektu
1. Zaimplementowanie gry Snake.
2. Stworzenie agenta AI (DQN) zdolnego do nauki.
3. Przeprowadzenie treningu i analiza postępów.
4. Wizualizacja procesu decyzyjnego agenta.

## Technologie
- Python 3.9.21
- Pygame 2.6.1
- TensorFlow 2.10.1 (Keras 2.10.0)
- NumPy 1.26.4

## Struktura Projektu
```
.
├── old scores/
│   ├── snake_dqn_model.weights2.h5
│   └── snake_dqn_training_log_continued_viz.csv
├── training_plots_with_avg/      # Katalog z wygenerowanymi wykresami
│   └── ... (pliki .png)
├── snake_game_ml.py              # Główny skrypt (trening, wizualizacja)
├── dqn_agent.py                  # Definicja agenta DQN
├── plot_training_log_separated_with_avg.py # Skrypt do generowania wykresów
├── requirements.txt              # Zależności projektu
├── README.md
```

## Instalacja i Uruchomienie
### Wymagania Wstępne
- Python 3.9+
- (Opcjonalnie dla GPU) Karta NVIDIA, CUDA, cuDNN.

### Kroki Instalacji
1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/GrabowskiB/PythonSnakeProject.git
   cd PythonSnakeProject
   ```
2. Utwórz i aktywuj środowisko wirtualne (np. conda):
   ```bash
   conda create -n snake_env python=3.9
   conda activate snake_env
   ```
3. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```
   *Upewnij się, że NumPy jest w wersji < 2.0 (np. 1.26.4) dla TensorFlow 2.10.x.*

### Uruchamianie
Skonfiguruj zmienne `VISUALIZE_TRAINING`, `LOAD_MODEL`, `MODEL_FILENAME`, `START_EPISODE_FROM_LOAD`, `target_global_episodes` w pliku `snake_game_ml.py` odpowiednio do potrzeb (trening, kontynuacja, wizualizacja).
```bash
python snake_game_ml.py
```
Dla wizualizacji ustaw `VISUALIZE_TRAINING = True` i niski `fps` (np. 10).

## Agent DQN
- **Architektura Sieci:** Wejście (11) -> Ukryta (128, ReLU) -> Wyjście (3, Liniowa).
- **Stan:** 11-bitowy wektor (kolizje, kierunek, pozycja jedzenia).
- **Akcje:** Prosto, Skręć w Lewo, Skręć w Prawo.
- **Nagrody:** +10 (jedzenie), -100 (kolizja), 0 (krok).
- **Hiperparametry:** $\gamma=0.95$, $\epsilon_{start}=1.0 \to \epsilon_{min}=0.01$ ($\epsilon_{decay}=0.995$), LR=0.001, Batch=64, Pamięć=20000.

## Wyniki Treningu
![Avg Score Plot](old_scores/training_plots_with_avg/plot_score_max_avg.png) 
*Średni wynik agenta (100 epizodów) w trakcie treningu. Pełna analiza w sprawozdaniu.*

Agent wykazał znaczną poprawę, osiągając średni wynik ok. 15 po 800 epizodach. Szczegółowe wykresy metryk znajdują się w katalogu `training_plots_with_avg/` (generowane przez `plot_training_log_separated_with_avg.py`).

## Możliwe Ulepszenia
- Dłuższy trening.
- Bardziej złożona reprezentacja stanu (np. CNN).
- Zaawansowane algorytmy RL (DDQN, Dueling DQN).

## Autor
**Bartek Grabowski**
- GitHub: [GrabowskiB](https://github.com/GrabowskiB)

## Licencja
Kod udostępniony do celów edukacyjnych.
