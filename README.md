# AutoML Project - Refaktoryzacja + Streamlit

## Opis projektu
Projekt obejmuje implementację modelu regresji liniowej służącego do przewidywania wartości na podstawie danych wejściowych oraz mechanizm ponownego trenowania modelu po dodaniu nowych obserwacji.

Dodatkowo zawiera aplikację webową stworzoną w Streamlit, która umożliwia korzystanie z modelu.

## Wymagania
```bash
pip install -r requirements.txt
```

## Uruchomienie modelu
```bash
python train_model.py
```

## Uruchomienie aplikacji Streamlit
```bash
streamlit run app/streamlit_app.py
```

## Funkcjonalności
### Refaktoryzacja
- predict_value(x) – przewiduje wartość y
- update_model(x, y) – dopisuje nowe dane i trenuje model ponownie

