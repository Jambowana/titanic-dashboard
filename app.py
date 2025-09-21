import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import process_data_and_train_model # Import our function

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Dashboard",
    page_icon="🚢",
    layout="wide"
)

# --- Data Loading and Model Training ---
# @st.cache_data runs this function only once to improve performance
@st.cache_data
def load_data_and_model():
    train_df, test_df, model, _ = process_data_and_train_model('train.csv', 'test.csv')
    return train_df, model

train_df, model = load_data_and_model()

# --- Sidebar for Filters ---
st.sidebar.header("Filter Passenger Data")

# Create a filter for Passenger Class
selected_class = st.sidebar.multiselect(
    "Passenger Class",
    options=train_df["Pclass"].unique(),
    default=train_df["Pclass"].unique()
)

# Create a filter for Sex
selected_sex = st.sidebar.multiselect(
    "Sex",
    options=train_df["Sex"].unique(), # 0 for male, 1 for female
    default=train_df["Sex"].unique()
)

# Filter the dataframe based on selection
df_selection = train_df.query(
    "Pclass == @selected_class & Sex == @selected_sex"
)

# --- Main Page ---
st.title("🚢 Titanic Survival Analysis Dashboard")
st.markdown("This dashboard provides an interactive analysis of the factors that influenced survival on the Titanic.")

# --- Key Metrics ---
total_passengers = df_selection.shape[0]
survival_rate = round(df_selection["Survived"].mean() * 100, 1)
survivors = df_selection["Survived"].sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Passengers in Selection", f"{total_passengers}")
with col2:
    st.metric("Total Survivors", f"{survivors}")
with col3:
    st.metric("Survival Rate", f"{survival_rate}%")

st.markdown("---")

# --- Interactive Charts ---
st.header("Visual Analysis of Survival")

# Chart 1: Survival Rate by Passenger Class
fig_class, ax_class = plt.subplots()
sns.barplot(data=df_selection, x="Pclass", y="Survived", ax=ax_class)
ax_class.set_title("Survival Rate by Passenger Class")
ax_class.set_ylabel("Survival Rate")

# Chart 2: Survival Rate by Sex
fig_sex, ax_sex = plt.subplots()
sns.barplot(data=df_selection, x="Sex", y="Survived", ax=ax_sex)
ax_sex.set_title("Survival Rate by Sex")
ax_sex.set_xticks([0, 1])
ax_sex.set_xticklabels(['Male', 'Female'])
ax_sex.set_ylabel("Survival Rate")

# Display charts side-by-side
left_column, right_column = st.columns(2)
left_column.pyplot(fig_class)
right_column.pyplot(fig_sex)


# --- Prediction Tool in Sidebar ---
st.sidebar.header("Check Your Survival Chance!")

# Input fields for user
pclass_input = st.sidebar.selectbox("Your Class", [1, 2, 3])
sex_input = st.sidebar.selectbox("Your Sex", ["Male", "Female"])
agegroup_input = st.sidebar.selectbox("Your Age Group", ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior'])
# Add other features if your model uses them, like SibSp, Parch, etc.

if st.sidebar.button("Predict"):
    # Convert text inputs to the numbers our model expects
    sex_num = 1 if sex_input == "Female" else 0
    age_map = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
    agegroup_num = age_map[agegroup_input]
    
    # Create a dataframe from the inputs for prediction
    # NOTE: The columns must match the ones used for training!
    # We will make up dummy values for the other columns.
    features = pd.DataFrame({
        'Pclass': [pclass_input],
        'Sex': [sex_num],
        'SibSp': [0], # Dummy value
        'Parch': [0], # Dummy value
        'Embarked': [1], # Dummy value
        'AgeGroup': [agegroup_num],
        'Title': [1], # Dummy value
        'FareBand': [1] # Dummy value
    })
    
    # Make prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    # Display result
    if prediction[0] == 1:
        st.sidebar.success(f"You Would Have Likely Survived! (Chance: {round(prediction_proba[0][1]*100, 2)}%)")
    else:
        st.sidebar.error(f"You Would Have Likely Not Survived. (Chance: {round(prediction_proba[0][1]*100, 2)}%)")