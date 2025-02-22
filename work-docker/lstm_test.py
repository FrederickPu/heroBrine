import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
def load_sequences_from_csv(filename, sequence_length=10):
  # Load dataset
  data = pd.read_csv(filename)  # Replace with your dataset path

  # Fill missing values
  # data['province'] = data['province'].fillna('unknown')
  #data['city'] = data['city'].fillna('unknown')

  # Encode categorical variables
  categorical_columns = ['debit_credit']
  for col in categorical_columns:
      data[col] = LabelEncoder().fit_transform(data[col])

  # Normalize numerical variables
  scaler = MinMaxScaler()
  data['amount_cad'] = scaler.fit_transform(data[['amount_cad']])

  # Combine transaction_date and transaction_time into a single datetime feature
  if 'transaction_time' in data.columns:
    data['transaction_datetime'] = pd.to_datetime(data['transaction_date'] + ' ' + data['transaction_time'])
    data = data.sort_values(by=['customer_id', 'transaction_datetime'])
  else:
    data = data.sort_values(by=['customer_id', 'transaction_date'])

  # Drop unused columns
  columns_to_keep = ['customer_id', 'amount_cad', 'debit_credit']
  data = data[columns_to_keep]

  # Group by customer_id and create sequences
  grouped = data.groupby('customer_id')
  sequences = []
  for customer_id, group in grouped:
      group = group.drop(columns=['customer_id']).values
      for i in range(len(group) - sequence_length + 1):
          sequences.append(group[i:i+sequence_length])
  sequences = np.array(sequences)
  return sequences

if __name__ == '__main__':

    sequence_length = 10
    sequences = load_sequences_from_csv("mnt/data/abm.csv", sequence_length)
    
    # Train-test split
    X_train, X_test = train_test_split(sequences, test_size=0.2, random_state=42)

    # LSTM Autoencoder Model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(sequence_length, X_train.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=False),
        tf.keras.layers.RepeatVector(sequence_length),
        tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X_train.shape[2]))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Train the model
    history = model.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.2, shuffle=True)

    # Reconstruction errors
    X_test_pred = model.predict(X_test)
    test_loss = np.mean(np.power(X_test - X_test_pred, 2), axis=(1, 2))

    # Anomaly detection threshold
    threshold = np.percentile(test_loss, 95)  # Adjust as needed
    anomalies = test_loss > threshold

    print(f"Threshold: {threshold}")
    print(f"Number of anomalies detected: {np.sum(anomalies)}")