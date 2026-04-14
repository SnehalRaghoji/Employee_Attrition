# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import FunctionTransformer

# # ✅ define drop columns used during training
# drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']

# # ✅ custom function
# def binary_cleanup(data):
#     data = data.copy()
#     data['Gender'] = data['Gender'].map({'Male':1,'Female':0})
#     data['OverTime'] = data['OverTime'].map({'Yes':1,'No':0})
#     return data.drop(columns=drop_cols)

# # Load model
# model = joblib.load('attrition_model.joblib')

# st.title("Employee Attrition Prediction App")

# # Inputs
# age = st.number_input("Age", 18, 60, 30)
# business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
# daily_rate = st.number_input("Daily Rate", value=500)
# department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
# distance_from_home = st.number_input("Distance From Home", value=5)
# education = st.selectbox("Education", [1,2,3,4,5])
# education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])

# employee_count = 1
# employee_number = 1
# over18 = 'Y'

# monthly_rate = st.number_input("Monthly Rate", value=10000)

# environment_satisfaction = st.selectbox("Environment Satisfaction", [1,2,3,4])
# gender = st.selectbox("Gender", ["Male", "Female"])
# hourly_rate = st.number_input("Hourly Rate", value=50)
# job_involvement = st.selectbox("Job Involvement", [1,2,3,4])
# job_level = st.selectbox("Job Level", [1,2,3,4,5])
# job_role = st.selectbox("Job Role", ["Sales Executive","Research Scientist","Laboratory Technician","Manager","Manufacturing Director","Healthcare Representative","Sales Representative","Research Director","Human Resources"])
# job_satisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])
# marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
# monthly_income = st.number_input("Monthly Income", value=5000)
# num_companies_worked = st.number_input("Companies Worked", value=1)
# over_time = st.selectbox("OverTime", ["Yes", "No"])
# percent_salary_hike = st.number_input("Salary Hike %", value=10)
# performance_rating = st.selectbox("Performance Rating", [1,2,3,4])
# relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1,2,3,4])
# stock_option_level = st.selectbox("Stock Option Level", [0,1,2,3])
# total_working_years = st.number_input("Total Working Years", value=5)
# training_times_last_year = st.number_input("Training Times", value=2)
# work_life_balance = st.selectbox("Work Life Balance", [1,2,3,4])
# years_at_company = st.number_input("Years At Company", value=3)
# years_in_current_role = st.number_input("Years In Role", value=2)
# years_since_last_promotion = st.number_input("Years Since Promotion", value=1)
# years_with_curr_manager = st.number_input("Years With Manager", value=2)
# standard_hours = 80

# if st.button("Predict"):
#     data = {
#         'Age': age,
#         'BusinessTravel': business_travel,
#         'DailyRate': daily_rate,
#         'Department': department,
#         'DistanceFromHome': distance_from_home,
#         'Education': education,
#         'EducationField': education_field,
#         'EmployeeCount': employee_count,
#         'EmployeeNumber': employee_number,
#         'EnvironmentSatisfaction': environment_satisfaction,
#         'Gender': gender,
#         'HourlyRate': hourly_rate,
#         'JobInvolvement': job_involvement,
#         'JobLevel': job_level,
#         'JobRole': job_role,
#         'JobSatisfaction': job_satisfaction,
#         'MaritalStatus': marital_status,
#         'MonthlyIncome': monthly_income,
#         'MonthlyRate': monthly_rate,
#         'NumCompaniesWorked': num_companies_worked,
#         'Over18': over18,
#         'OverTime': over_time,
#         'PercentSalaryHike': percent_salary_hike,
#         'PerformanceRating': performance_rating,
#         'RelationshipSatisfaction': relationship_satisfaction,
#         'StandardHours': standard_hours,
#         'StockOptionLevel': stock_option_level,
#         'TotalWorkingYears': total_working_years,
#         'TrainingTimesLastYear': training_times_last_year,
#         'WorkLifeBalance': work_life_balance,
#         'YearsAtCompany': years_at_company,
#         'YearsInCurrentRole': years_in_current_role,
#         'YearsSinceLastPromotion': years_since_last_promotion,
#         'YearsWithCurrManager': years_with_curr_manager
#     }

#     df = pd.DataFrame([data])

#     # Predictions
#     pred = model.predict(df)[0]
#     probs = model.predict_proba(df)[0]

#     prob_no = probs[0]
#     prob_yes = probs[1]

#     st.subheader("Prediction Result")
#     if pred == 1:
#         st.error("Attrition: YES")
#     else:
#         st.success("Attrition: NO")

#     st.write(f"Probability (No): {prob_no:.2f}")
#     st.write(f"Probability (Yes): {prob_yes:.2f}")

#     # ✅ Top 2 Important Features
#     try:
#         feature_names = model.named_steps['preprocessing'].get_feature_names_out()
#         coefficients = model.named_steps['model'].coef_[0]

#         feat_imp = pd.DataFrame({
#             'Feature': feature_names,
#             'Importance': coefficients
#         })

#         feat_imp['Abs'] = feat_imp['Importance'].abs()
#         top_features = feat_imp.sort_values(by='Abs', ascending=False).head(2)

#         st.subheader("Top 2 Important Features")
#         st.write(top_features[['Feature', 'Importance']])

#         fig, ax = plt.subplots()
#         ax.barh(top_features['Feature'], top_features['Importance'])
#         ax.set_title("Top 2 Features Impact")
#         st.pyplot(fig)

#     except Exception:
#         st.warning("Feature importance not available.")


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Attrition Predictor", layout="wide")

# -------------------- PREPROCESSING --------------------
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']

def binary_cleanup(data):
    data = data.copy()
    data['Gender'] = data['Gender'].map({'Male':1,'Female':0})
    data['OverTime'] = data['OverTime'].map({'Yes':1,'No':0})
    return data.drop(columns=drop_cols)

# -------------------- LOAD MODEL --------------------
model = joblib.load('attrition_model.joblib')

# -------------------- UI --------------------
st.title("📊 Employee Attrition Prediction Dashboard")
st.markdown("### Fill employee details to predict attrition risk")

# Layout using columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 60, 30)
    daily_rate = st.number_input("Daily Rate", value=500)
    distance_from_home = st.slider("Distance From Home", 1, 50, 5)
    monthly_income = st.number_input("Monthly Income", value=5000)
    monthly_rate = st.number_input("Monthly Rate", value=10000)

with col2:
    job_level = st.selectbox("Job Level", [1,2,3,4,5])
    job_role = st.selectbox("Job Role", ["Sales Executive","Research Scientist","Laboratory Technician","Manager","Manufacturing Director","Healthcare Representative","Sales Representative","Research Director","Human Resources"])
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    work_life_balance = st.slider("Work Life Balance", 1, 4, 3)

with col3:
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    education = st.selectbox("Education", [1,2,3,4,5])
    gender = st.radio("Gender", ["Male", "Female"])
    over_time = st.radio("OverTime", ["Yes", "No"])

# Additional inputs
education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_companies_worked = st.slider("Companies Worked", 0, 10, 1)
percent_salary_hike = st.slider("Salary Hike %", 0, 50, 10)
performance_rating = st.selectbox("Performance Rating", [1,2,3,4])
relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
stock_option_level = st.selectbox("Stock Option Level", [0,1,2,3])
total_working_years = st.slider("Total Working Years", 0, 40, 5)
training_times_last_year = st.slider("Training Times", 0, 10, 2)
years_at_company = st.slider("Years At Company", 0, 40, 3)
years_in_current_role = st.slider("Years In Role", 0, 20, 2)
years_since_last_promotion = st.slider("Years Since Promotion", 0, 15, 1)
years_with_curr_manager = st.slider("Years With Manager", 0, 20, 2)

# Constants
employee_count = 1
employee_number = 1
over18 = 'Y'
standard_hours = 80
hourly_rate = 50
job_involvement = 3

# -------------------- PREDICTION --------------------
if st.button("🚀 Predict Attrition"):
    data = {
        'Age': age,
        'BusinessTravel': business_travel,
        'DailyRate': daily_rate,
        'Department': department,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EducationField': education_field,
        'EmployeeCount': employee_count,
        'EmployeeNumber': employee_number,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'Over18': over18,
        'OverTime': over_time,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StandardHours': standard_hours,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }

    df = pd.DataFrame([data])

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    prob_no, prob_yes = probs[0], probs[1]

    st.subheader("📌 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        if pred == 1:
            st.error("Attrition: YES")
        else:
            st.success("Attrition: NO")

    with colB:
        st.metric("Probability (No)", f"{prob_no:.2f}")
        st.metric("Probability (Yes)", f"{prob_yes:.2f}")

    # -------------------- FEATURE IMPORTANCE --------------------
    try:
        feature_names = model.named_steps['preprocessing'].get_feature_names_out()
        coefficients = model.named_steps['model'].coef_[0]

        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefficients
        })

        feat_imp['Abs'] = feat_imp['Importance'].abs()
        top_features = feat_imp.sort_values(by='Abs', ascending=False).head(2)

        st.subheader("🔥 Top 2 Influential Features")

        fig, ax = plt.subplots()
        ax.barh(top_features['Feature'], top_features['Importance'])
        ax.set_xlabel("Impact")
        ax.set_ylabel("Feature")
        st.pyplot(fig)

    except Exception:
        st.warning("Feature importance not available.")

st.markdown("---")
st.caption("💡 Built with Streamlit | ML Attrition Model")

