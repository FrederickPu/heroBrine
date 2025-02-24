import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def process_csv(filename):
  # Load dataset
  data = pd.read_csv(filename)  

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

  # Keep only necessary columns
  columns_to_keep = ['customer_id', 'amount_cad', 'debit_credit']
  data = data[columns_to_keep]

  return data


def load_sequences_from_csv(data, sequence_length=10):
  # Group by customer_id and create sequences
  grouped = data.groupby('customer_id')
  customer_ids = []
  data_vectors = []
  sequences = []
  for customer_id, group in grouped:
    group = group.drop(columns=['customer_id']).values
    for i in range(len(group) - sequence_length + 1):
      sequences.append(group[i:i+sequence_length])
      if customer_id not in customer_ids:
         customer_ids.append(customer_id)
         data_vectors.append(group[i:i+sequence_length])
  sequences = np.array(sequences)
  return customer_ids, data_vectors, sequences

if __name__ == '__main__':
  sequence_length = 75
  model_path = "model_lstm.h5"
  output_file = "/mnt/output/customer_embeddings.txt"
  data_dir = "/mnt/data/"

  # Load sequences
  processed_csvs = []
  for file in os.listdir(data_dir):
    if file.endswith(".csv"):
      file_path = os.path.join(data_dir, file)
      processed_csvs.append(process_csv(file_path))
  data = pd.concat(processed_csvs, ignore_index=True)
  customer_ids, data_vectors, sequences = load_sequences_from_csv(data, sequence_length)
  
  # Train-test split
  X_train, X_test = train_test_split(sequences, test_size=0.2, random_state=42)

  # Define model architecture
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

  # Load pre-trained weights if available
  if os.path.exists(model_path):
      print(f"Loading model weights from {model_path}...")
      model.load_weights(model_path)
  else:
      print(f"No pre-trained model found at {model_path}. Please train and save the model.")

  # Predict anomalies
  X_test_pred = model.predict(X_test)
  test_loss = np.mean(np.power(X_test - X_test_pred, 2), axis=(1, 2))

  # Anomaly detection threshold
  threshold = np.percentile(test_loss, 95)
  anomalies = test_loss > threshold

  print(f"Threshold: {threshold}")
  print(f"Number of anomalies detected: {np.sum(anomalies)}")
  
  embeddings = model.predict(np.array(data_vectors))

  visited = set()
  with open(output_file, "w") as f:
    for customer_id, embedding in zip(customer_ids, embeddings):
      if customer_id not in visited:
        f.write(f"{customer_id}," + ",".join(map(str, embedding)) + "\n")
        visited.add(customer_id)
    f.write("\n")  # Ensure an empty line at the end