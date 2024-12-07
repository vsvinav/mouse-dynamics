import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    user_data = {}
    user_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found user directories: {user_dirs}")

    for user_dir in user_dirs:
        user_id = user_dir
        user_path = os.path.join(data_dir, user_dir)
        csv_files = glob.glob(os.path.join(user_path, "*.csv")) + glob.glob(os.path.join(user_path, "*.CSV"))
        print(f"User {user_id} CSV files: {csv_files}")
        sequences = []

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            sequence = preprocess_sequence(df)
            sequences.append(sequence)

        user_data[user_id] = sequences

    return user_data


def preprocess_sequence(df):
    # Convert timestamp to time differences
    epsilon = 1e-6
    df['time_diff'] = df['client timestamp'].diff().fillna(0) + epsilon

    # Normalize x and y coordinates
    df['x'] = df['x'] / df['x'].max()
    df['y'] = df['y'] / df['y'].max()

    # Calculate velocities
    df['velocity'] = np.sqrt(df['x'].diff() ** 2 + df['y'].diff() ** 2) / df['time_diff']
    df['velocity'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['velocity'].fillna(0, inplace=True)

    # Calculate direction (angle)
    df['direction'] = np.arctan2(df['y'].diff(), df['x'].diff())
    df['direction'].fillna(0, inplace=True)

    # Select features
    features = df[['x', 'y', 'velocity', 'direction']].values

    # Handle NaN or Inf
    if np.isnan(features).any() or np.isinf(features).any():
        print("Warning: NaN or Inf values detected in features.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad or truncate to fixed length
    max_seq_length = 100
    if len(features) < max_seq_length:
        pad_width = max_seq_length - len(features)
        features = np.pad(features, ((0, pad_width), (0, 0)), 'constant')
    else:
        features = features[:max_seq_length]

    return features


def create_pairs(user_data):
    pairs = []
    labels = []
    users = list(user_data.keys())

    for user in users:
        sequences = user_data[user]
        num_sequences = len(sequences)
        if num_sequences < 2:
            print(f"Not enough sequences for user {user} to create positive pairs.")
            continue
        # Positive pairs
        for i in range(num_sequences):
            for j in range(i + 1, num_sequences):
                pairs.append([sequences[i], sequences[j]])
                labels.append(1)  # Similar (same user)

        # Negative pairs
        other_users = [u for u in users if u != user]
        for other_user in other_users:
            other_sequences = user_data[other_user]
            if len(other_sequences) == 0:
                continue
            for seq in other_sequences:
                pairs.append([sequences[np.random.randint(num_sequences)], seq])
                labels.append(0)  # Dissimilar (different users)

    print(f"Total pairs created: {len(pairs)}")
    return np.array(pairs), np.array(labels)



def create_embedding_network(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(128, activation='relu'))
    return model

def create_siamese_network(input_shape):
    # Inputs
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    # Embedding network
    embedding_network = create_embedding_network(input_shape)

    # Get embeddings
    embedding_a = embedding_network(input_a)
    embedding_b = embedding_network(input_b)

    # Compute (embedding_a - embedding_b)
    diff = layers.Subtract()([embedding_a, embedding_b])
    # Compute squared difference: diff^2
    squared_diff = layers.Multiply()([diff, diff])
    # Sum over all features to get L2 distance squared:
    # Dot product of diff with itself sums the element-wise product
    distance = layers.Dot(axes=1)([diff, diff])

    # Optional: Add Dense layers to convert distance into a similarity score
    # Note: The network can learn to interpret this distance directly
    x = layers.Dense(64, activation='relu')(distance)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    siamese_model = models.Model(inputs=[input_a, input_b], outputs=outputs)
    return siamese_model, embedding_network

# Training

# Load the data
data_dir = 'random-data'  # Replace with your data directory
user_data = load_data(data_dir)

# Debug print
print(f"Loaded data for {len(user_data)} users.")
for user_id, sequences in user_data.items():
    print(f"User {user_id} has {len(sequences)} sequences.")

# Create pairs and labels
pairs, labels = create_pairs(user_data)

# Split into training/validation
pairs_train, pairs_val, labels_train, labels_val = train_test_split(pairs, labels, test_size=0.2, random_state=42)
x1_train = np.array([pair[0] for pair in pairs_train])
x2_train = np.array([pair[1] for pair in pairs_train])
x1_val = np.array([pair[0] for pair in pairs_val])
x2_val = np.array([pair[1] for pair in pairs_val])

# Define input shape
input_shape = x1_train.shape[1:]

# Create the Siamese network
siamese_model, embedding_network = create_siamese_network(input_shape)

# Compile
optimizer = optimizers.Adam(learning_rate=1e-4)
siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = siamese_model.fit([x1_train, x2_train], labels_train,
                            validation_data=([x1_val, x2_val], labels_val),
                            batch_size=32, epochs=50,
                            callbacks=[early_stopping])

# Save the embedding network
embedding_network.save('embedding_network.h5')

# Load the embedding network
# embedding_network = tf.keras.models.load_model('embedding_network.h5')



def authenticate_user(new_sequence_df, registered_sequences, threshold=0.5):
    # Preprocess the new sequence
    new_sequence = preprocess_sequence(new_sequence_df)
    new_sequence = np.expand_dims(new_sequence, axis=0)

    # Compute embedding for the new sequence
    new_embedding = embedding_network.predict(new_sequence)

    # Compare with registered embeddings
    similarities = []
    for seq in registered_sequences:
        seq = np.expand_dims(seq, axis=0)
        registered_embedding = embedding_network.predict(seq)
        distance = np.linalg.norm(new_embedding - registered_embedding)
        similarities.append(distance)

    avg_similarity = np.mean(similarities)
    is_authenticated = avg_similarity < threshold

    return avg_similarity, is_authenticated