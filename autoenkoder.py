import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Wczytanie danych z pliku users.json
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



# Podział danych na zbiór treningowy i testowy (80% trening, 20% test)
X_train, X_test = train_test_split(interests_data, test_size=0.2, random_state=42)
print(f"Zbiór treningowy: {X_train.shape}")
print(f"Zbiór testowy: {X_test.shape}")



# Budowa modelu autoenkodera
input_dim = 46
latent_dim = 2

# Warstwa wejściowa
input_layer = Input(shape=(input_dim,))

# Koder
encoded = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
encoded = Dropout(0.1)(encoded)
encoded = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(encoded)
encoded = Dropout(0.1)(encoded)
encoded = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(encoded)
encoded = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(encoded)
latent = Dense(latent_dim, activation=None, name='latent')(encoded)

# Dekoder
decoded = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(latent)
decoded = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(decoded)
decoded = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(decoded)
decoded = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

# Model autoenkodera
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Model kodera
encoder = Model(inputs=input_layer, outputs=latent)

# Wagi dla funkcji straty
class_weight = {0: 1.0, 1: (total_values / total_ones) * 1.0}
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

# Podsumowanie modelu
autoencoder.summary()



# Trening modelu
history = autoencoder.fit(
    X_train, X_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, X_test),
    verbose=1
)



# Testowanie modelu na zbiorze testowym
# Przewidywanie na zbiorze testowym
reconstructed = autoencoder.predict(X_test)

# Zaokrąglenie wartości wyjściowych do 0 lub 1
reconstructed_binary = (reconstructed > 0.3).astype(int)

# Obliczenie dokładności odtworzenia
accuracy = np.mean((reconstructed_binary == X_test).astype(int), axis=1)
mean_accuracy = np.mean(accuracy)
print(f"Średnia dokładność odtworzenia: {mean_accuracy * 100:.2f}%")

# Analiza błędów
false_positives = np.sum((reconstructed_binary == 1) & (X_test == 0))  # 0 błędnie jako 1
false_negatives = np.sum((reconstructed_binary == 0) & (X_test == 1))  # 1 błędnie jako 0
print(f"Liczba błędów (0 jako 1): {false_positives}")
print(f"Liczba błędów (1 jako 0): {false_negatives}")



# Testowanie na trzech różnych przykładach
print("\n=== Testowanie na trzech przykładach ===")

example_1 = np.array([[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 0, 0, 0, 1, 0]], dtype=np.float32)

example_2 = np.array([[0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 1, 1, 0, 0, 0]], dtype=np.float32)

example_3 = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 1, 1, 0, 1]], dtype=np.float32)

# Testowanie kodera
latent_1 = encoder.predict(example_1)
latent_2 = encoder.predict(example_2)
latent_3 = encoder.predict(example_3)

# Testowanie autoenkodera
recon_1 = autoencoder.predict(example_1)
recon_2 = autoencoder.predict(example_2)
recon_3 = autoencoder.predict(example_3)

# Zaokrąglenie rekonstrukcji do wartości binarnych
recon_1_binary = (recon_1 > 0.3).astype(int)
recon_2_binary = (recon_2 > 0.3).astype(int)
recon_3_binary = (recon_3 > 0.3).astype(int)

# Wyświetlenie wyników dla każdego przykładu
print("\nPrzykład 1:")
print(f"Oryginalny wektor: {example_1}")
print(f"Wartości warstwy ukrytej: {latent_1[0]}")
print(f"Odtworzony wektor: {recon_1_binary[0]}")
print(f"Dokładność odtworzenia: {np.mean(example_1 == recon_1_binary) * 100:.2f}%")

print("\nPrzykład 2:")
print(f"Oryginalny wektor: {example_2}")
print(f"Wartości warstwy ukrytej: {latent_2[0]}")
print(f"Odtworzony wektor: {recon_2_binary[0]}")
print(f"Dokładność odtworzenia: {np.mean(example_2 == recon_2_binary) * 100:.2f}%")

print("\nPrzykład 3:")
print(f"Oryginalny wektor: {example_3}")
print(f"Wartości warstwy ukrytej: {latent_3[0]}")
print(f"Odtworzony wektor: {recon_3_binary[0]}")
print(f"Dokładność odtworzenia: {np.mean(example_3 == recon_3_binary) * 100:.2f}%")



# Wizualizacja straty treningowej i walidacyjnej
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.title('Strata modelu w trakcie treningu')
plt.xlabel('Epoka')
plt.ylabel('Strata (binary cross-entropy)')
plt.legend()
plt.show()



# Zapis modelu w formacie Keras
encoder.save('encoder_model.keras')
print("Model kodera zapisany jako 'encoder_model.keras'")

autoencoder.save('autoencoder_model.keras')
print("Model autoenkodera zapisany jako 'autoencoder_model.keras'")



# Konwersja modeli do formatu TFLite
converter_encoder = tf.lite.TFLiteConverter.from_keras_model(encoder)
converter_encoder.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_encoder_model = converter_encoder.convert()

with open('encoder_model.tflite', 'wb') as f:
    f.write(tflite_encoder_model)
print("Model kodera zapisany jako 'encoder_model.tflite'")

# Konwersja autoenkodera
converter_autoencoder = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
converter_autoencoder.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_autoencoder_model = converter_autoencoder.convert()

with open('autoencoder_model.tflite', 'wb') as f:
    f.write(tflite_autoencoder_model)
print("Model autoenkodera zapisany jako 'autoencoder_model.tflite'")