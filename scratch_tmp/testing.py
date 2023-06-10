import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout

X = np.stack(df_filtered['Lake_data_1D'].to_numpy())

df_filtered['is_snow_precip'] = df_filtered['is_snow_precip'].apply(lambda x: int(round(x)) if isinstance(x, float) and not np.isnan(x) else (int(x) if not np.isnan(x) else 0))

y = df_filtered['is_snow_precip'].values.astype(int)
# print(y)

# Fill NaN values with 0
X = np.nan_to_num(X)
y = np.nan_to_num(y)


input_data = []
output_data = []

for i in range(len(X) - 120):
    input_data.append(X[i:i+72])
    output_data.append(y[i+120])

input_data = np.stack(input_data)
output_data = np.stack(output_data)


# Scale the input data
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data.reshape(input_data.shape[0], -1)).reshape(input_data.shape)

# Reshape the input data to match Conv1D input shape (batch_size, steps, input_dim)
input_data_scaled = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], input_data_scaled.shape[2])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data_scaled, output_data, test_size=0.2, random_state=42)


model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Conv1D(64, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae', 'accuracy'])


# model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy'])


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

