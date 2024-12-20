import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset dengan data aktivitas harian dan tidur cukup
data = {
    'Wake_Up_Time': [6, 7, 8, 6, 5],  # Waktu bangun (jam)
    'Study_Time': [3, 2, 4, 1, 3],  # Durasi belajar (jam)
    'Exercise_Time': [1, 1.5, 2, 1, 0.5],  # Durasi olahraga (jam)
    'Phone_Time': [2, 3, 1, 4, 2],  # Waktu bermain HP (jam)
    'Sleep_Duration': [7, 6, 7.5, 5.5, 6],  # Durasi tidur (jam)
    'Adequate_Sleep': [1, 0, 1, 0, 0]  # 1 jika tidur cukup, 0 jika tidak cukup
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Memisahkan fitur dan target
X = df.drop(columns=['Adequate_Sleep', 'Sleep_Duration'])  # Fitur
y = df['Adequate_Sleep']  # Target (1 = tidur cukup, 0 = tidak cukup)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat dan melatih model Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Fungsi untuk menerima inputan pengguna dan memprediksi apakah tidur cukup
def predict_sleep():
    print("Masukkan data aktivitas harian Anda:")
    
    # Inputan pengguna
    wake_up_time = int(input("Masukkan waktu bangun (jam): "))
    study_time = float(input("Masukkan durasi belajar (jam): "))
    exercise_time = float(input("Masukkan durasi olahraga (jam): "))
    phone_time = float(input("Masukkan waktu bermain HP (jam): "))
    
    # Membuat array input untuk prediksi
    user_input = [[wake_up_time, study_time, exercise_time, phone_time]]
    
    # Memprediksi apakah tidur cukup (1) atau tidak (0)
    prediction = rf.predict(user_input)
    
    if prediction[0] == 1:
        print("Anda tidur cukup!")
    else:
        print("Anda tidak tidur cukup!")

# Menjalankan fungsi untuk prediksi input pengguna
predict_sleep()
