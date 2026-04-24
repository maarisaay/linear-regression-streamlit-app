import streamlit as st
from transformers import pipeline, MarianMTModel, MarianTokenizer

st.title("Aplikacja NLP - analiza tekstu i tłumaczenie")

st.write(
    "Aplikacja umożliwia analizę wydźwięku tekstu w języku angielskim "
    "oraz tłumaczenie tekstu z angielskiego na niemiecki."
)

option = st.selectbox(
    "Wybierz funkcję:",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie tekstu z angielskiego na niemiecki",
    ],
)

text = st.text_area("Wpisz tekst:")

if text:
    try:
        with st.spinner("Przetwarzanie..."):

            if option == "Wydźwięk emocjonalny tekstu (eng)":
                classifier = pipeline("sentiment-analysis")
                result = classifier(text)
                st.success("Analiza zakończona.")
                st.write(result)

            elif option == "Tłumaczenie tekstu z angielskiego na niemiecki":
                model_name = "Helsinki-NLP/opus-mt-en-de"

                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                inputs = tokenizer(text, return_tensors="pt", padding=True)
                translated = model.generate(**inputs)
                translation = tokenizer.decode(translated[0], skip_special_tokens=True)

                st.success("Tłumaczenie zakończone.")
                st.write(translation)

    except Exception as e:
        st.error("Wystąpił błąd podczas działania aplikacji.")
        st.write(e)

st.write("Numer indeksu: s27375")