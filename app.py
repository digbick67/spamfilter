import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Enkel data för träning (baserat på din notebook)
data = {
    'text': [
        'Jag älskar denna produkt, den är fantastisk!', 
        'Detta är det sämsta jag någonsin köpt.',
        'Helt okej, men leveransen var sen.',
        'Fantastisk kundservice och bra kvalitet.',
        'Riktigt dålig upplevelse, rekommenderas inte.'
    ],
    'label': ['positiv', 'negativ', 'neutral', 'positiv', 'negativ']
}

df = pd.DataFrame(data)

# Skapa modell-pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Träna modellen
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Textklassificering - Analys")
user_input = st.text_input("Skriv en text för att analysera sentiment:")

if user_input:
    prediction = model.predict([user_input])
    st.write(f"Modellen klassificerar detta som: **{prediction[0]}**")
