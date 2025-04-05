import json
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Wczytanie danych z pliku users.json
with open('users.json', 'r', encoding='utf-8') as file:
    users_data = json.load(file)

# Wyodrębnienie wektorów zainteresowań
interests_data = np.array([user['interests'] for user in users_data])
print(f"Liczba użytkowników: {len(interests_data)}")
print(f"Przykład wektora zainteresowań: {interests_data[0]}")

# Analiza balansu danych (ile jest wartości 1 w wektorach)
total_ones = np.sum(interests_data)
total_values = interests_data.size
print(f"Procent wartości 1 w danych: {total_ones / total_values * 100:.2f}%")

# 2. Podział danych na zbiór treningowy i testowy (80% trening, 20% test)
X_train, X_test = train_test_split(interests_data, test_size=0.2, random_state=42)
print(f"Zbiór treningowy: {X_train.shape}")
print(f"Zbiór testowy: {X_test.shape}")

# 3. Budowa modelu autoenkodera
input_dim = 46  # Liczba zainteresowań
latent_dim = 2  # Warstwa ukryta (x, y)

# Warstwa wejściowa
input_layer = Input(shape=(input_dim,))

# Koder
encoded = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
encoded = Dropout(0.1)(encoded)
encoded = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(encoded)
encoded = Dropout(0.1)(encoded)
encoded = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(encoded)
encoded = Dense(8, activation='relu', kernel_regularizer=l2(0.01))(encoded)
latent = Dense(latent_dim, activation='sigmoid', name='latent')(encoded)  # Warstwa ukryta (x, y)

# Dekoder
decoded = Dense(8, activation='relu', kernel_regularizer=l2(0.01))(latent)
decoded = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(decoded)
#decoded = Dropout(0.1)(decoded)
decoded = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(decoded)
#decoded = Dropout(0.1)(decoded)
decoded = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)  # Warstwa wyjściowa

# Model autoenkodera
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Model kodera (do wyciągania x, y)
encoder = Model(inputs=input_layer, outputs=latent)

# Wagi dla funkcji straty (zwiększona waga dla wartości 1)
class_weight = {0: 1.0, 1: (total_values / total_ones) * 2.2}  # Większa waga dla wartości 1
autoencoder.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy')

# Podsumowanie modelu
autoencoder.summary()

# 4. Trening modelu
history = autoencoder.fit(
    X_train, X_train,
    epochs=1000,
    batch_size=16,
    validation_data=(X_test, X_test),
    verbose=1
)

# 5. Testowanie modelu
# Przewidywanie na zbiorze testowym
reconstructed = autoencoder.predict(X_test)

# Zaokrąglenie wartości wyjściowych do 0 lub 1 (próg 0.3 zamiast 0.5)
reconstructed_binary = (reconstructed > 0.3).astype(int)

# Obliczenie dokładności odtworzenia
accuracy = np.mean((reconstructed_binary == X_test).astype(int), axis=1)
mean_accuracy = np.mean(accuracy)
print(f"Średnia dokładność odtworzenia: {mean_accuracy * 100:.2f}%")

# Analiza błędów (ile razy model pomylił 0 na 1 i 1 na 0)
false_positives = np.sum((reconstructed_binary == 1) & (X_test == 0))  # 0 błędnie jako 1
false_negatives = np.sum((reconstructed_binary == 0) & (X_test == 1))  # 1 błędnie jako 0
print(f"Liczba błędów (0 jako 1): {false_positives}")
print(f"Liczba błędów (1 jako 0): {false_negatives}")

# 6. Wyciągnięcie wartości x, y dla pierwszego użytkownika w zbiorze testowym
latent_values = encoder.predict(X_test)
print("\nPrzykład dla pierwszego użytkownika w zbiorze testowym:")
print(f"Oryginalny wektor: {X_test[0]}")
print(f"Odtworzony wektor: {reconstructed_binary[0]}")
print(f"Wartości x, y (warstwa ukryta): {latent_values[0]}")

# 7. Wizualizacja straty treningowej i walidacyjnej
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.title('Strata modelu w trakcie treningu')
plt.xlabel('Epoka')
plt.ylabel('Strata (binary cross-entropy)')
plt.legend()
plt.show()

# 8. Zapis modelu w formacie Keras
encoder.save('encoder_model.keras')
print("Model kodera zapisany jako 'encoder_model.keras'")

autoencoder.save('autoencoder_model.keras')
print("Model zapisany jako 'autoencoder_model.keras'")