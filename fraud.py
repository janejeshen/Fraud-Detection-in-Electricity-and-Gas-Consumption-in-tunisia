import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl
import base64
import warnings
warnings.filterwarnings('error')
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

st.set_page_config(page_title='Tunisia Electricity Fraud Detection', page_icon='🕵️', layout="wide", initial_sidebar_state="auto")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
   <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('/home/jane/Documents/electricity/Fraud-Detection-in-Electricity-and-Gas-Consumption-in-tunisia/soroush-zargar-Nu-QDChzGEw-unsplash.jpg')

# st.markdown(
#     """
#     <style>
#     .big-font {
#         font-size: 16px !important;
#         font-family: sans-serif;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# Use the CSS class to apply the font to a text element
# st.markdown('<p class="big-font">⚠️This is not the official Tunisian Company of Electricity and Gas app</p>', unsafe_allow_html=True)



st.header('Detecting Electricity Fraud In Tunisia')

st.markdown('The Tunisian Company of Electricity and Gas, STEG, experienced large losses as a result of consumer fraud. My app analyzes usage patterns and flags suspect accounts using data modeling to find potential fraudsters in order to stop further losses.')
st.write('##### ⚠️This is not the official Tunisian Company of Electricity and Gas app‼️')
# Load your data

data = pd.read_csv('/home/jane/Documents/electricity/Fraud-Detection-in-Electricity-and-Gas-Consumption-in-tunisia/data/agg_train.csv')

# Identify categorical features
cat_features = ['district', 'client_catg', 'region']

# One-hot encode categorical features
encoder = OneHotEncoder()
encoded = encoder.fit_transform(data[cat_features])

# Combine encoded features with numerical features
numerical_features = ['transactions_count', 'consommation_level_1_mean', 'consommation_level_2_mean', 'consommation_level_3_mean', 'consommation_level_4_mean']
X = pd.concat([pd.DataFrame(encoded.toarray()), data[numerical_features]], axis=1)

# Load the saved model using the load_model() function
def load_model(path):
    with open(path, 'rb') as f:
        fraud_model = pkl.load(f)
    return fraud_model

fraud_model = load_model('/home/jane/Documents/electricity/Fraud-Detection-in-Electricity-and-Gas-Consumption-in-tunisia/model.pkl')

#Dummy function for the model prediction
def predict(district, client_catg, region, transactions_count, consommation_level_1_mean, consommation_level_2_mean, consommation_level_3_mean, consommation_level_4_mean):
    # Insert your model prediction code here
    # This is just a dummy function that returns the sum of the input features
    return transactions_count + consommation_level_1_mean + consommation_level_2_mean + consommation_level_3_mean + consommation_level_4_mean

# User input for features
district = st.selectbox('District', data['district'].unique())
client_catg = st.selectbox('Client Category', data['client_catg'].unique())
region = st.selectbox('Region', data['region'].unique())
transactions_count = st.number_input('Transactions Count', value=1, step=1)
consommation_level_1_mean = st.number_input('Consommation Level 1 Mean', value=0.0, step=0.1)
consommation_level_2_mean = st.number_input('Consommation Level 2 Mean', value=0.0, step=0.1)
consommation_level_3_mean = st.number_input('Consommation Level 3 Mean', value=0.0, step=0.1)
consommation_level_4_mean = st.number_input('Consommation Level 4 Mean', value=0.0, step=0.1)

input_data = pd.DataFrame([[district, client_catg, region, transactions_count, consommation_level_1_mean, consommation_level_2_mean, consommation_level_3_mean, consommation_level_4_mean]], columns=['district', 'client_catg', 'region', 'transactions_count', 'consommation_level_1_mean', 'consommation_level_2_mean', 'consommation_level_3_mean', 'consommation_level_4_mean'])

encoded_input = encoder.transform(input_data[cat_features])
input_features = pd.concat([pd.DataFrame(encoded_input.toarray()), input_data[numerical_features]], axis=1)
prediction = fraud_model.predict(input_features.values)



# button_style = 'background-color: green; color: white; font-weight: bold;'
if st.button("Predict", key='predict_button', help='Click here to make a prediction',use_container_width=True):
    with st.spinner("Making prediction..."):
        # Use the trained model to predict the outcome and display it to the user
        if prediction[0] == 0:
            st.write('*Prediction:* The customer is not engaging in fraudulent activities.')
        else:
            st.write('*Prediction:* The customer is engaging in fraudulent activities.')
    st.success("Prediction completed!")







st.markdown('<div class="contact-container">', unsafe_allow_html=True)

with st.container():
    st.write("### Contact Info")
    st.write("Email: janenjuguna550@gmail.com")
    st.write("Phone: +254114180510")
    st.write("Address: Nairobi,Kenya")
    st.write("Github: https://github.com/janejeshen")

    st.markdown(
        """
        <style>
        .contact-container {
            background-color: #d1c1aa;
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
# Close the div tag
st.markdown('</div>', unsafe_allow_html=True)