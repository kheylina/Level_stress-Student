import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv('student_lifestyle_dataset.csv')

# Mapping stress levels
stress_mapping = {
    'Low': 0,
    'Moderate': 1,
    'High': 2
}
df['Stress_Level'] = df['Stress_Level'].map(stress_mapping)

# Split data for model training
X = df[['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']]
y = df['Stress_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Streamlit app
st.title('Prediksi Tingkat Stres Mahasiswa')

# Input features
study_hours = st.number_input('Jam Belajar per Hari', min_value=0.0, max_value=24.0, value=5.0)
extracurricular_hours = st.number_input('Jam Ekstrakurikuler per Hari', min_value=0.0, max_value=24.0, value=1.0)
sleep_hours = st.number_input('Jam Tidur per Hari', min_value=0.0, max_value=24.0, value=7.0)
social_hours = st.number_input('Jam Sosial per Hari', min_value=0.0, max_value=24.0, value=2.0)
physical_activity_hours = st.number_input('Jam Aktivitas Fisik per Hari', min_value=0.0, max_value=24.0, value=1.0)
gpa = st.number_input('GPA', min_value=0.0, max_value=4.0, value=3.0)

# Predict button
if st.button('Prediksi'):
    # Prepare data for prediction
    input_data = np.array([[study_hours, extracurricular_hours, sleep_hours, social_hours, physical_activity_hours, gpa]])
    prediction = knn.predict(input_data)
    
    # Map prediction back to stress level
    stress_levels = {0: 'Low', 1: 'Moderate', 2: 'High'}
    stress_level = stress_levels[prediction[0]]
    st.write(f'Tingkat Stres: {stress_level}')

# Visualize stress level distribution
st.subheader('Distribusi Level Stres')
stress_counts = df['Stress_Level'].value_counts()
stress_percentages = df['Stress_Level'].value_counts(normalize=True) * 100
stress_summary = pd.DataFrame({'Count': stress_counts, 'Percentage': stress_percentages})

# Plot bar chart
fig, ax = plt.subplots()
colors = ['blue', 'green', 'red']
stress_counts.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
ax.set_title('Distribusi Level Stres')
ax.set_xlabel('Level Stres')
ax.set_ylabel('Jumlah')
for i, (count, percentage) in enumerate(zip(stress_counts, stress_percentages)):
    ax.text(i, count + 2, f'{count} ({percentage:.1f}%)', ha='center', color=colors[i])

st.pyplot(fig)

# Visualize average hours per activity
st.subheader('Rata-rata Waktu yang Dihabiskan per Aktivitas')
rata2_extracurricular = df['Extracurricular_Hours_Per_Day'].mean()
rata2_sleep = df['Sleep_Hours_Per_Day'].mean()
rata2_social = df['Social_Hours_Per_Day'].mean()
data_rata2 = pd.DataFrame({
    'Aktivitas': ['Extracurricular', 'Sleep', 'Social'],
    'Rata-rata Jam': [rata2_extracurricular, rata2_sleep, rata2_social]
})

fig2, ax2 = plt.subplots()
sns.barplot(x='Aktivitas', y='Rata-rata Jam', data=data_rata2, ax=ax2)
ax2.set_title('Perbandingan Rata-rata Waktu yang Dihabiskan')
ax2.set_xlabel('Aktivitas')
ax2.set_ylabel('Rata-rata Jam per Hari')

st.pyplot(fig2)


