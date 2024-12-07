import tkinter as tk
from tkinter import messagebox
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import random
import os

######################################
# Utility Functions
######################################

def preprocess_sequence(df):
    epsilon = 1e-6
    df['time_diff'] = df['client timestamp'].diff().fillna(0) + epsilon

    if df['x'].max() != 0:
        df['x'] = df['x'] / df['x'].max()
    if df['y'].max() != 0:
        df['y'] = df['y'] / df['y'].max()

    df['velocity'] = np.sqrt(df['x'].diff()**2 + df['y'].diff()**2) / df['time_diff']
    df['velocity'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['velocity'].fillna(0, inplace=True)

    df['direction'] = np.arctan2(df['y'].diff(), df['x'].diff())
    df['direction'].fillna(0, inplace=True)

    features = df[['x', 'y', 'velocity', 'direction']].values
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    max_seq_length = 100
    if len(features) < max_seq_length:
        pad_width = max_seq_length - len(features)
        features = np.pad(features, ((0, pad_width), (0, 0)), 'constant')
    else:
        features = features[:max_seq_length]

    return features

def authenticate_user(new_sequence, registered_sequence, embedding_network, threshold=0.5):
    if embedding_network is None:
        return float('inf'), False
    if registered_sequence is None:
        return float('inf'), False

    new_sequence = np.expand_dims(new_sequence, axis=0)
    reg_sequence = np.expand_dims(registered_sequence, axis=0)

    new_embedding = embedding_network.predict(new_sequence)
    registered_embedding = embedding_network.predict(reg_sequence)

    distance = np.linalg.norm(new_embedding - registered_embedding)
    is_authenticated = distance < threshold
    return distance, is_authenticated


######################################
# Main Application
######################################

class MouseDynamicsApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Mouse Dynamics Demo")

        # Load the embedding network model if available
        self.embedding_network = None
        if os.path.exists('embedding_network.h5'):
            try:
                self.embedding_network = models.load_model('embedding_network.h5')
            except Exception as e:
                messagebox.showwarning("Model Warning", f"Failed to load model: {e}")
        else:
            messagebox.showinfo("No Model", "No 'embedding_network.h5' found. Authentication may fail.")

        # Create UI
        self.frame = tk.Frame(self.master)
        self.frame.pack(pady=20)

        self.enroll_button = tk.Button(self.frame, text="Enroll User", command=self.start_enrollment)
        self.enroll_button.pack(side='left', padx=10)

        self.auth_button = tk.Button(self.frame, text="Authenticate User", command=self.start_authentication)
        self.auth_button.pack(side='left', padx=10)

        # Variables to store and handle game states
        self.game_window = None

    def start_enrollment(self):
        if self.game_window is not None and tk.Toplevel.winfo_exists(self.game_window):
            messagebox.showerror("Error", "Game already in progress!")
            return
        self.game_window = tk.Toplevel(self.master)
        self.game_window.title("Enrollment Game")
        EnrollmentGame(self.game_window, self.embedding_network)

    def start_authentication(self):
        if not os.path.exists('user_data.csv'):
            messagebox.showerror("No Enrollment Data", "No 'user_data.csv' found. Please enroll a user first.")
            return
        if self.game_window is not None and tk.Toplevel.winfo_exists(self.game_window):
            messagebox.showerror("Error", "Game already in progress!")
            return
        self.game_window = tk.Toplevel(self.master)
        self.game_window.title("Authentication Game")
        AuthenticationGame(self.game_window, self.embedding_network)


######################################
# Enrollment Game Class (Improved Initial Game)
######################################

class EnrollmentGame:
    def __init__(self, master, embedding_network):
        self.master = master
        self.embedding_network = embedding_network

        # Instead of a static target, we will show a random target at different positions
        # User must hit the target multiple times, each time it appears at a random location.
        self.required_clicks = 10
        self.target_clicks = 0
        self.mouse_data = []

        self.instruction_label = tk.Label(self.master, text="ENROLLMENT: Move your mouse around and click the red circle 10 times.\nThe circle will appear at random places each hit.")
        self.instruction_label.pack(pady=10)

        self.canvas = tk.Canvas(self.master, width=800, height=600, bg='white')
        self.canvas.pack()

        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)

        self.target = None
        self.place_new_target()

    def place_new_target(self):
        if self.target is not None:
            self.canvas.delete(self.target)
        # Random position ensures the user moves the mouse to find it
        x = random.randint(100, 700)
        y = random.randint(100, 500)
        r = 30
        self.target = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='red')

    def on_mouse_move(self, event):
        timestamp = int(time.time() * 1000)
        self.mouse_data.append([timestamp, 'NoButton', 'Move', event.x, event.y])

    def on_mouse_down(self, event):
        items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        if self.target in items:
            timestamp = int(time.time() * 1000)
            self.mouse_data.append([timestamp, 'Button1', 'Down', event.x, event.y])
            self.target_clicks += 1
            if self.target_clicks < self.required_clicks:
                messagebox.showinfo("Hit!", f"Good job! Hits so far: {self.target_clicks}")
                self.place_new_target()
            else:
                # Enrollment complete
                messagebox.showinfo("Enrollment Complete", "You have clicked the target 10 times. Saving data...")
                self.save_data_and_close()

    def save_data_and_close(self):
        df = pd.DataFrame(self.mouse_data, columns=['client timestamp', 'button', 'state', 'x', 'y'])
        df.to_csv('user_data.csv', index=False)
        messagebox.showinfo("Data Saved", "Your movement data has been saved to 'user_data.csv'.")
        self.master.destroy()


######################################
# Authentication Game Class
######################################

class AuthenticationGame:
    def __init__(self, master, embedding_network):
        self.master = master
        self.embedding_network = embedding_network

        # Similar game, but maybe smaller targets to differentiate?
        # The user must click 5 times.
        self.required_clicks = 5
        self.target_clicks = 0
        self.mouse_data = []

        self.instruction_label = tk.Label(self.master, text="AUTHENTICATION: Click the smaller red circle 5 times.\nIt will appear at random places each hit. Let's see if you match your enrolled pattern!")
        self.instruction_label.pack(pady=10)

        self.canvas = tk.Canvas(self.master, width=800, height=600, bg='white')
        self.canvas.pack()

        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)

        self.target = None
        self.place_new_target()

    def place_new_target(self):
        if self.target is not None:
            self.canvas.delete(self.target)
        # Random position again, smaller radius
        x = random.randint(100, 700)
        y = random.randint(100, 500)
        r = 20
        self.target = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='red')

    def on_mouse_move(self, event):
        timestamp = int(time.time() * 1000)
        self.mouse_data.append([timestamp, 'NoButton', 'Move', event.x, event.y])

    def on_mouse_down(self, event):
        items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        if self.target in items:
            timestamp = int(time.time() * 1000)
            self.mouse_data.append([timestamp, 'Button1', 'Down', event.x, event.y])
            self.target_clicks += 1
            if self.target_clicks < self.required_clicks:
                messagebox.showinfo("Hit!", f"Target hit! Hits so far: {self.target_clicks}")
                self.place_new_target()
            else:
                # Authentication attempt
                messagebox.showinfo("Task Complete", "5 hits done. Authenticating...")
                self.authenticate()

    def authenticate(self):
        # Load enrolled user data
        if not os.path.exists('user_data.csv'):
            messagebox.showerror("No Data", "No 'user_data.csv' found. Cannot authenticate.")
            self.master.destroy()
            return

        reg_df = pd.read_csv('user_data.csv')
        registered_sequence = preprocess_sequence(reg_df)

        df = pd.DataFrame(self.mouse_data, columns=['client timestamp', 'button', 'state', 'x', 'y'])
        new_sequence = preprocess_sequence(df)

        distance, authenticated = authenticate_user(new_sequence, registered_sequence, self.embedding_network)

        if self.embedding_network is None:
            messagebox.showerror("No Model", "No embedding model available, cannot authenticate.")
        else:
            if authenticated:
                messagebox.showinfo("Authenticated", f"User authenticated successfully! Distance: {distance:.4f}")
            else:
                messagebox.showerror("Authentication Failed", f"User not authenticated. Distance: {distance:.4f}")

        self.master.destroy()


######################################
# Run the Application
######################################

def main():
    root = tk.Tk()
    app = MouseDynamicsApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
