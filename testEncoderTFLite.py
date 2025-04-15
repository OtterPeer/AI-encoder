import numpy as np
import tensorflow as tf

# Wczytaj model TFLite
interpreter = tf.lite.Interpreter(model_path='encoder_model.tflite')
interpreter.allocate_tensors()

# Pobierz szczegóły wejścia i wyjścia
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Przygotuj dane wejściowe
input_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 1]], dtype=np.float32)

# Ustaw dane wejściowe
interpreter.set_tensor(input_details[0]['index'], input_data)

# Uruchom inferencję
interpreter.invoke()

# Pobierz dane wyjściowe
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Wartości x, y:", output_data)