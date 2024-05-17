import re
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from collections import Counter
import operator
import joblib
import streamlit as st
import base64
import time  # Import the time module for sleep
import nltk

nltk.data.path.append('nltk_data')
warnings.filterwarnings("ignore")

# Initialize NLP tools
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Function to add background image
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load the trained Logistic Regression model using joblibs
lr_comb = joblib.load("model.joblib")
add_bg_from_local("bg.png")  # Adjust the path if needed
df_comb = pd.read_csv("dis_sym_dataset_comb.csv")  # Disease combination
df_norm = pd.read_csv("dis_sym_dataset_norm.csv")  # Individual Disease
doctors = pd.read_csv("doctors.csv")

dataset_symptoms = list(df_norm.columns[1:])
doctors['Specialization'] = doctors['Specialization'].apply(lambda x: [spec.strip() for spec in x.split(',')])
doctors['Normalized Satisfaction Score'] = doctors['Patient Satisfaction Rate(%age)'] * doctors['Total_Reviews']

# Mapping diseases to medical specializations
disease_to_specialization = {
    # ... (same dictionary as provided)
}

def diseaseDetail(term):
    query = f"{term} site:wikipedia.org"
    results = []
    try:
        for url in search(query, num_results=10, stop=10, pause=2):
            if 'wikipedia.org' in url:
                results.append(url)
                break  # Only take the first Wikipedia result
    except Exception as e:
        return f"Error occurred during search: {e}"

    if not results:
        return f"No Wikipedia page found for {term}."

    wiki_url = results[0]
    try:
        wiki_response = requests.get(wiki_url)
        wiki_response.raise_for_status()
        soup = BeautifulSoup(wiki_response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        details = ' '.join([para.get_text() for para in paragraphs[:3]])
        return details
    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error: {errh}"
    except requests.exceptions.ConnectionError as errc:
        return f"Error Connecting: {errc}"
    except requests.exceptions.Timeout as errt:
        return f"Timeout Error: {errt}"
    except requests.exceptions.RequestException as err:
        return f"An Error Occurred: {err}"

def synonyms(term):
    synonyms = []
    response = requests.get(f'https://www.thesaurus.com/browse/{term}')
    soup = BeautifulSoup(response.content, "html.parser")
    try:
        container = soup.find('section', {'class': 'MainContentContainer'})
        row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'}).find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        pass
    for syn in wordnet.synsets(term):
        synonyms += syn.lemma_names()
    return set(synonyms)

def get_specialization(disease_name):
    return disease_to_specialization.get(disease_name, "Specialization not found")

# Initialize Streamlit app
st.title("Disease Prediction and Doctor Recommendation System")

# Add custom CSS for text color
st.markdown(
    """
    <style>
    .stApp {{
        color: #333333;  /* Set text color to a dark shade for better contrast */
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #333333 !important;  /* Set headings to the same color */
    }}
    .stTextInput > div > div > input {{
        color: #333333 !important;  /* Set input text color */
    }}
    .stButton > button {{
        color: #333333 !important;  /* Set button text color */
    }}
    .stMarkdown p {{
        color: #333333 !important;  /* Set markdown text color */
    }}
    .stCheckbox > div {{
        color: #333333 !important;  /* Set checkbox text color */
    }}
    .stSelectbox > div > div > div > div {{
        color: #333333 !important;  /* Set selectbox text color */
    }}
    .stRadio > div {{
        color: #333333 !important;  /* Set radio button text color */
    }}
    .css-1d391kg p {{
        color: #333333 !important;  /* Set other paragraph text color */
    }}
    .css-2trqyj p {{
        color: #333333 !important;  /* Set paragraph text color */
    }}
    .css-10trblm {{
        color: #333333 !important;  /* Set other text color */
    }}
    .css-1gk0hsc {{
        color: #333333 !important;  /* Set text color */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# User input for symptoms
if 'step' not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    user_input = st.text_input("Enter symptoms separated by comma:")
    if st.button("Submit Symptoms"):
        user_symptoms = user_input.lower().split(',')
        st.session_state.user_symptoms = [sym.strip() for sym in user_symptoms]
        st.session_state.step = 2

if st.session_state.step == 2:
    # Preprocess the input symptoms
    processed_user_symptoms = []
    for sym in st.session_state.user_symptoms:
        sym = sym.replace('-', ' ').replace("'", '')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)

    # Query expansion
    expanded_symptoms = []
    for user_sym in processed_user_symptoms:
        user_sym = user_sym.split()
        str_sym = set()
        for comb in range(1, len(user_sym) + 1):
            for subset in combinations(user_sym, comb):
                subset = ' '.join(subset)
                subset = synonyms(subset)
                str_sym.update(subset)
        str_sym.add(' '.join(user_sym))
        expanded_symptoms.append(' '.join(str_sym).replace('_', ' '))

    found_symptoms = set()
    for data_sym in dataset_symptoms:
        data_sym_split = data_sym.split()
        for user_sym in expanded_symptoms:
            count = 0
            for symp in data_sym_split:
                if symp in user_sym.split():
                    count += 1
            if count / len(data_sym_split) > 0.5:
                found_symptoms.add(data_sym)
    found_symptoms = list(found_symptoms)

    st.session_state.found_symptoms = found_symptoms
    st.write("Top matching symptoms from your search:")
    for idx, symp in enumerate(found_symptoms):
        st.write(f"{idx}: {symp}")
    
    selected_symptoms = st.multiselect("Select relevant symptoms:", found_symptoms)
    if st.button("Submit Selected Symptoms"):
        st.session_state.selected_symptoms = selected_symptoms
        st.session_state.step = 3

if st.session_state.step == 3:
    final_symp = st.session_state.selected_symptoms
    dis_list = set()
    counter_list = []
    for symp in final_symp:
        dis_list.update(set(df_norm[df_norm[symp] == 1]['label_dis']))

    for dis in dis_list:
        row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
        row[0].pop(0)
        for idx, val in enumerate(row[0]):
            if val != 0 and dataset_symptoms[idx] not in final_symp:
                counter_list.append(dataset_symptoms[idx])

    dict_symp = dict(Counter(counter_list))
    dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1), reverse=True)

    found_symptoms = []
    for idx, (symptom, count) in enumerate(dict_symp_tup[:5]):
        found_symptoms.append(symptom)
    st.write("Common co-occurring symptoms:")
    for idx, symp in enumerate(found_symptoms):
        st.write(f"{idx}: {symp}")
    
    selected_additional_symptoms = st.multiselect("Do you have any of these symptoms?", found_symptoms)
    if st.button("Submit Additional Symptoms"):
        final_symp.extend(selected_additional_symptoms)
        st.session_state.final_symp = final_symp
        st.session_state.step = 4

if st.session_state.step == 4:
    final_symp = st.session_state.final_symp
    sample_x = [0 for _ in range(len(dataset_symptoms))]
    for val in final_symp:
        sample_x[dataset_symptoms.index(val)] = 1

    prediction = lr_comb.predict_proba([sample_x])
    topk = prediction[0].argsort()[-10:][::-1]

    most_probable_diseases = []
    st.write("Top Diseases Predicted:")
    for idx in topk:
        most_probable_diseases.append(df_comb['label_dis'].unique()[idx])
        st.write(f"{df_comb['label_dis'].unique()[idx]}")

    st.session_state.most_probable_diseases = most_probable_diseases
    st.session_state.step = 5

if st.session_state.step == 5:
    select_disease = st.selectbox("Select a disease to see details:", st.session_state.most_probable_diseases)
    if select_disease:
        details = diseaseDetail(select_disease)
        st.write(f"### {select_disease}")
        st.write(details)
    
    if st.button("Show Doctor Recommendations"):
        st.session_state.selected_disease = select_disease
        st.session_state.step = 6

if st.session_state.step == 6:
    disease = st.session_state.selected_disease
    specialization_needed = get_specialization(disease)
    st.write(f"Required Specialization: {specialization_needed}")

    filtered_doctors = doctors[doctors['Specialization'].apply(lambda x: specialization_needed in x)]
    top_doctor = filtered_doctors.sort_values(by='Normalized Satisfaction Score', ascending=False).head(1)

    if not top_doctor.empty:
        doctor = top_doctor.iloc[0]
        st.write(f"Recommended Doctor: {doctor['Doctor Name']}")
        st.write(f"City: {doctor['City']}")
        st.write(f"Specialization: {doctor['Specialization']}")
        st.write(f"Qualification: {doctor['Doctor Qualification']}")
        st.write(f"Experience: {doctor['Experience(Years)']} years")
        st.write(f"Reviews: {doctor['Total_Reviews']}")
        st.write(f"Satisfaction Rate: {doctor['Patient Satisfaction Rate(%age)']}%")
        st.write(f"Average Time to Patients: {doctor['Avg Time to Patients(mins)']} mins")
        st.write(f"Wait Time: {doctor['Wait Time(mins)']} mins")
        st.write(f"Fee: PKR {doctor['Fee(PKR)']}")
        st.write(f"Hospital Address: {doctor['Hospital Address']}")
        st.write(f"Profile Link: {doctor['Doctors Link']}")
    else:
        st.write("No available doctors for this specialization.")
