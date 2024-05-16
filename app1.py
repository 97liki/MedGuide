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

# Load the trained Logistic Regression model using joblib
lr_comb = joblib.load("C:\\Users\\Likitha\\Downloads\\docotr_deploy\\model.joblib")
add_bg_from_local("C:\\Users\\Likitha\\Downloads\\docotr_deploy\\bg.png")  # Adjust the path if needed
df_comb = pd.read_csv("C:\\Users\\Likitha\\Downloads\\docotr_deploy\\Dataset\\dis_sym_dataset_comb.csv")  # Disease combination
df_norm = pd.read_csv("C:\\Users\\Likitha\\Downloads\\docotr_deploy\\Dataset\\dis_sym_dataset_norm.csv")  # Individual Disease
doctors = pd.read_csv("C:\\Users\\Likitha\\Downloads\\docotr_deploy\\doctors.csv")


dataset_symptoms = list(df_norm.columns[1:])
doctors['Specialization'] = doctors['Specialization'].apply(lambda x: [spec.strip() for spec in x.split(',')])
doctors['Normalized Satisfaction Score'] = doctors['Patient Satisfaction Rate(%age)'] * doctors['Total_Reviews']

# Mapping diseases to medical specializations
disease_to_specialization = {
    'Abscess': 'General Surgeon',
    'Acquired Capillary Haemangioma of Eyelid': 'Dermatologist',
    'Acquired Immuno Deficiency Syndrome': 'Infectious Diseases',
    'Acute encephalitis syndrome': 'Neurologist',
    'Adult Inclusion Conjunctivitis': 'Eye Specialist',
    'Alcohol Abuse and Alcoholism': 'Psychiatrist',
    'Alopecia (hair loss)': 'Dermatologist',
    'Alzheimer': 'Neurologist',
    'Amaurosis Fugax': 'Ophthalmologist',
    'Amblyopia': 'Ophthalmologist',
    'Amoebiasis': 'Gastroenterologist',
    'Anaemia': 'Hematologist',
    'Aniseikonia': 'Ophthalmologist',
    'Anisometropia': 'Ophthalmologist',
    'Antepartum hemorrhage (Bleeding in late pregnancy)': 'Gynecologist',
    'Anthrax': 'Infectious Diseases',
    'Anxiety': 'Psychiatrist',
    'Appendicitis': 'General Surgeon',
    'Arthritis': 'Rheumatologist',
    'Asbestos-related diseases': 'Pulmonologist / Lung Specialist',
    'Aseptic meningitis': 'Neurologist',
    'Asthma': 'Pulmonologist / Lung Specialist',
    'Astigmatism': 'Ophthalmologist',
    'Atrophy': 'Neurologist',
    'Autism': 'Pediatrician',
    'Bad Breath (Halitosis)': 'General Physician',
    "Bell's Palsy": 'Neurologist',
    'Beriberi': 'Nutritionist',
    'Black Death': 'Infectious Diseases',
    'Bleeding Gums': 'Dentist',
    'Blindness': 'Ophthalmologist',
    'Botulism': 'Infectious Diseases',
    'Brain Tumour': 'Neuro Surgeon',
    'Breast Cancer / Carcinoma': 'Oncologist',
    'Bronchitis': 'Pulmonologist / Lung Specialist',
    'Brucellosis': 'Infectious Diseases',
    'Bubonic plague': 'Infectious Diseases',
    'Bunion': 'Orthopedic Surgeon',
    'Burns': 'General Surgeon',
    'Calculi': 'Urologist',
    'Campylobacter infection': 'Gastroenterologist',
    'Cancer': 'Oncologist',
    'Candidiasis': 'Infectious Diseases',
    'Carbon monoxide poisoning': 'Emergency Medicine',
    'Carpal Tunnel Syndrome': 'Orthopedic Surgeon',
    'Cavities': 'Dentist',
    'Celiacs disease': 'Gastroenterologist',
    'Cerebral palsy': 'Pediatric Neurologist',
    'Chagas disease': 'Infectious Diseases',
    'Chalazion': 'Ophthalmologist',
    'Chickenpox': 'Pediatrician',
    'Chikungunya Fever': 'Infectious Diseases',
    'Childhood Exotropia': 'Ophthalmologist',
    'Chlamydia': 'Sexologist',
    'Cholera': 'Infectious Diseases',
    'Chorea': 'Neurologist',
    'Chronic fatigue syndrome': 'Internal Medicine Specialist',
    'Chronic obstructive pulmonary disease (COPD)': 'Pulmonologist / Lung Specialist',
    'Cleft Lip and Cleft Palate': 'Plastic Surgeon',
    'Colitis': 'Gastroenterologist',
    'Colorectal Cancer': 'Oncologist',
    'Common cold': 'General Physician',
    'Condyloma': 'Dermatologist',
    'Congenital anomalies (birth defects)': 'Geneticist',
    'Congestive heart disease': 'Cardiologist',
    'Corneal Abrasion': 'Ophthalmologist',
    'Coronary Heart Disease': 'Cardiologist',
    'Coronavirus disease 2019 (COVID-19)': 'Infectious Diseases',
    'Cough': 'General Physician',
    'Crimean Congo haemorrhagic fever (CCHF)': 'Infectious Diseases',
    'Dehydration': 'General Physician',
    'Dementia': 'Neurologist',
    'Dengue': 'Infectious Diseases',
    'Diabetes Mellitus': 'Endocrinologist',
    'Diabetic Retinopathy': 'Ophthalmologist',
    'Diarrhea': 'Gastroenterologist',
    'Diphtheria': 'Infectious Diseases',
    "Down's Syndrome": 'Geneticist',
    'Dracunculiasis (guinea-worm disease)': 'Infectious Diseases',
    'Dysentery': 'Gastroenterologist',
    'Ear infection': 'Ent Specialist',
    'Early pregnancy loss': 'Gynecologist',
    'Ebola': 'Infectious Diseases',
    'Eclampsia': 'Gynecologist',
    'Ectopic pregnancy': 'Gynecologist',
    'Eczema': 'Dermatologist',
    'Endometriosis': 'Gynecologist',
    'Epilepsy': 'Neurologist',
    'Fibroids': 'Gynecologist',
    'Fibromyalgia': 'Rheumatologist',
    'Food Poisoning': 'Gastroenterologist',
    'Frost Bite': 'General Surgeon',
    'GERD': 'Gastroenterologist',
    'Gaming disorder': 'Psychiatrist',
    'Gangrene': 'General Surgeon',
    'Gastroenteritis': 'Gastroenterologist',
    'Genital herpes': 'Dermatologist',
    'Glaucoma': 'Ophthalmologist',
    'Goitre': 'Endocrinologist',
    'Gonorrhea': 'Sexologist',
    'Guillain-Barré syndrome': 'Neurologist',
    'Haemophilia': 'Hematologist',
    'Hand, Foot and Mouth Disease': 'Pediatrician',
    'Heat-Related Illnesses and Heat waves': 'Emergency Medicine',
    'Hepatitis': 'Hepatologist',
    'Hepatitis A': 'Hepatologist',
    'Hepatitis B': 'Hepatologist',
    'Hepatitis C': 'Hepatologist',
    'Hepatitis D': 'Hepatologist',
    'Hepatitis E': 'Hepatologist',
    'Herpes Simplex': 'Dermatologist',
    'High risk pregnancy': 'Gynecologist',
    'Human papillomavirus': 'Dermatologist',
    'Hypermetropia': 'Ophthalmologist',
    'Hyperthyroidism': 'Endocrinologist',
    'Hypothyroid': 'Endocrinologist',
    'Hypotonia': 'Pediatrician',
    'Impetigo': 'Dermatologist',
    'Inflammatory Bowel Disease': 'Gastroenterologist',
    'Influenza': 'General Physician',
    'Insomnia': 'Psychiatrist',
    'Interstitial cystitis': 'Urologist',
    'Iritis': 'Ophthalmologist',
    'Iron Deficiency Anemia': 'Hematologist',
    'Irritable bowel syndrome': 'Gastroenterologist',
    'Japanese Encephalitis': 'Infectious Diseases',
    'Jaundice': 'Hepatologist',
    'Kala-azar/ Leishmaniasis': 'Infectious Diseases',
    'Kaposi’s Sarcoma': 'Oncologist',
    'Keratoconjunctivitis Sicca (Dry eye syndrome)': 'Ophthalmologist',
    'Keratoconus': 'Ophthalmologist',
    'Kuru': 'Neurologist',
    'Laryngitis': 'Ent Specialist',
    'Lead poisoning': 'Toxicologist',
    'Legionellosis': 'Infectious Diseases',
    'Leprosy': 'Dermatologist',
    'Leptospirosis': 'Infectious Diseases',
    'Leukemia': 'Hematologist',
    'Lice': 'Dermatologist',
    'Lung cancer': 'Oncologist',
    'Lupus erythematosus': 'Rheumatologist',
    'Lyme disease': 'Infectious Diseases',
    'Lymphoma': 'Oncologist',
    'Mad cow disease': 'Neurologist',
    'Malaria': 'Infectious Diseases',
    'Marburg fever': 'Infectious Diseases',
    'Mastitis': 'Gynecologist',
    'Measles': 'Pediatrician',
    'Melanoma': 'Oncologist',
    'Middle East respiratory syndrome coronavirus (MERS‐CoV)': 'Infectious Diseases',
    'Migraine': 'Neurologist',
    'Mononucleosis': 'Infectious Diseases',
    'Mouth Breathing': 'Ent Specialist',
    'Multiple myeloma': 'Oncologist',
    'Multiple sclerosis': 'Neurologist',
    'Mumps': 'Pediatrician',
    'Muscular dystrophy': 'Neurologist',
    'Myasthenia gravis': 'Neurologist',
    'Myelitis': 'Neurologist',
    'Myocardial Infarction (Heart Attack)': 'Cardiologist',
    'Myopia': 'Ophthalmologist',
    'Narcolepsy': 'Neurologist',
    'Nasal Polyps': 'Ent Specialist',
    'Nausea and Vomiting of Pregnancy and  Hyperemesis gravidarum': 'Gynecologist',
    'Necrotizing Fasciitis': 'General Surgeon',
    'Neonatal Respiratory Disease Syndrome(NRDS)': 'Pediatrician',
    'Neoplasm': 'Oncologist',
    'Neuralgia': 'Neurologist',
    'Nipah virus infection': 'Infectious Diseases',
    'Obesity': 'Nutritionist',
    'Obsessive Compulsive Disorder': 'Psychiatrist',
    'Oral Cancer': 'Oncologist',
    'Orbital Dermoid': 'Ophthalmologist',
    'Osteoarthritis': 'Orthopedic Surgeon',
    'Osteomyelitis': 'Orthopedic Surgeon',
    'Osteoporosis': 'Orthopedic Surgeon',
    'Paratyphoid fever': 'Infectious Diseases',
    "Parkinson's Disease": 'Neurologist',
    'Pelvic inflammatory disease': 'Gynecologist',
    'Perennial Allergic Conjunctivitis': 'Allergy Specialist',
    'Pericarditis': 'Cardiologist',
    'Peritonitis': 'General Surgeon',
    'Pinguecula': 'Ophthalmologist',
    'Pneumonia': 'Pulmonologist / Lung Specialist',
    'Poliomyelitis': 'Pediatrician',
    'Polycystic ovary syndrome (PCOS)': 'Endocrinologist',
    'Porphyria': 'Dermatologist',
    'Post Menopausal Bleeding': 'Gynecologist',
    'Post-herpetic neuralgia': 'Pain Specialist',
    'Postpartum depression/ Perinatal depression': 'Psychiatrist',
    'Preeclampsia': 'Gynecologist',
    'Premenstrual syndrome': 'Gynecologist',
    'Presbyopia': 'Ophthalmologist',
    'Preterm birth': 'Gynecologist',
    'Progeria': 'Geneticist',
    'Psoriasis': 'Dermatologist',
    'Puerperal sepsis': 'Gynecologist',
    'Pulmonary embolism': 'Pulmonologist / Lung Specialist',
    'Ques fever': 'Infectious Diseases',
    'Quinsy': 'Ent Specialist',
    'Rabies': 'Infectious Diseases',
    "Raynaud's Phenomenon": 'Rheumatologist',
    'Repetitive strain injury': 'Orthopedic Surgeon',
    'Rheumatic fever': 'Cardiologist',
    'Rheumatism': 'Rheumatologist',
    'Rickets': 'Pediatrician',
    'Rift Valley fever': 'Infectious Diseases',
    'Rocky Mountain spotted fever': 'Infectious Diseases',
    'Rubella': 'Pediatrician',
    'SARS': 'Infectious Diseases',
    'SIDS': 'Pediatrician',
    'Sarcoidosis': 'Pulmonologist / Lung Specialist',
    'Sarcoma': 'Oncologist',
    'Scabies': 'Dermatologist',
    'Scarlet fever': 'Infectious Diseases',
    'Schizophrenia': 'Psychiatrist',
    'Sciatica': 'Orthopedic Surgeon',
    'Scrapie': 'Neurologist',
    'Scrub Typhus': 'Infectious Diseases',
    'Scurvy': 'Nutritionist',
    'Sepsis': 'Infectious Diseases',
    'Sexually transmitted infections (STIs)': 'Sexologist',
    'Shaken Baby Syndrome': 'Pediatrician',
    'Shigellosis': 'Infectious Diseases',
    'Shin splints': 'Orthopedic Surgeon',
    'Shingles': 'Dermatologist',
    'Sickle-cell anemia': 'Hematologist',
    'Smallpox': 'Infectious Diseases',
    'Stevens-Johnson syndrome': 'Dermatologist',
    'Stomach ulcers': 'Gastroenterologist',
    'Strep throat': 'Ent Specialist',
    'Stroke': 'Neurologist',
    'Sub-conjunctival Haemorrhage': 'Ophthalmologist',
    'Syphilis': 'Sexologist',
    'Taeniasis': 'Infectious Diseases',
    'Taeniasis/cysticercosis': 'Infectious Diseases',
    'Tay-Sachs disease': 'Geneticist',
    'Tennis elbow': 'Orthopedic Surgeon',
    'Tetanus': 'Infectious Diseases',
    'Thalassaemia': 'Hematologist',
    'Tinnitus': 'Ent Specialist',
    'Tonsillitis': 'Ent Specialist',
    'Toxic shock syndrome': 'Infectious Diseases',
    'Trachoma': 'Ophthalmologist',
    'Trichinosis': 'Infectious Diseases',
    'Trichomoniasis': 'Sexologist',
    'Tuberculosis': 'Pulmonologist / Lung Specialist',
    'Tularemia': 'Infectious Diseases',
    'Turners Syndrome': 'Geneticist',
    'Urticaria': 'Dermatologist',
    'Varicose Veins': 'Vascular Surgeon',
    'Vasovagal syncope': 'Cardiologist',
    'Vitamin B12 Deficiency': 'Nutritionist',
    'Vitiligo': 'Dermatologist',
    'Warkany syndrome': 'Geneticist',
    'Warts': 'Dermatologist',
    'Yaws': 'Dermatologist',
    'Yellow Fever': 'Infectious Diseases',
    'Zika virus disease': 'Infectious Diseases',
    'lactose intolerance': 'Gastroenterologist',
    'papilloedema': 'Ophthalmologist'
}

def diseaseDetail(term):
    diseases = [term]
    ret = ""
    for dis in diseases:
        query = dis + ' wikipedia'
        for sr in search(query, tld="co.in", stop=10, pause=0.5):
            match = re.search(r'wikipedia', sr)
            filled = 0
            if match:
                wiki = requests.get(sr, verify=False)
                soup = BeautifulSoup(wiki.content, 'html.parser')  # Use default parser
                info_table = soup.find("table", {"class": "infobox"})
                if info_table is not None:
                    for row in info_table.find_all("tr"):
                        data = row.find("th", {"scope": "row"})
                        if data is not None:
                            if "Pronunciation" in data.get_text():
                                continue
                            symptom = str(row.find("td"))
                            symptom = re.sub(r'<[^<]+?>', ' ', symptom)
                            symptom = re.sub(r'\[.*\]', '', symptom)
                            symptom = symptom.replace("&gt", ">").strip()
                            ret += f"{data.get_text()} - {symptom}\n\n"
                            filled = 1
                if filled:
                    break
    return ret

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
st.title("Disease Predictor and Doctor Recommender")

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

    # st.session_state.expanded_symptoms = expanded_symptoms
    # st.write("After query expansion, your symptoms are:")
    # st.write(expanded_symptoms)
    
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
    select_disease = st.selectbox("More details about the disease:", st.session_state.most_probable_diseases)
    if select_disease:
        st.write(diseaseDetail(select_disease))
    
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