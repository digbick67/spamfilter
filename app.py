import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Data för träning
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

# Skapa och träna modell
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)
model.fit(X_train, y_train)

# Webbgränssnitt
st.title("🤖 Textklassificering")
st.write("Skriv in en text nedan så gissar AI:n om den är positiv eller negativ.")

user_input = st.text_input("Din text:")
if user_input:
    prediction = model.predict([user_input])
    st.success(f"Resultat: {prediction[0]}")
