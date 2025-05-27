import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

print("--- Checking TensorFlow Configuration ---")
print("TensorFlow Version:", tf.__version__)
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("Found GPU:", gpu_devices)
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory Growth set for GPU.")
    except RuntimeError as e:
        print(e)
else:
    print("!!! No GPU found by TensorFlow. Training will run on CPU.")
print("-----------------------------------------")

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)

        # DQN Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.model = self._build_model()  # Main Q-network
        self.target_model = self._build_model()  # Target Q-network
        self.update_target_model()  # Synchronize weights

    def _build_model(self):
        # Simple neural network (MLP)
        model = models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')  # Output layer
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from the main model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action_index, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, action_index, reward, next_state, done))

    def act(self, state):
        # Choose action: exploration (random) or exploitation (model-based)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = np.array(state).reshape(1, -1)
        act_values = self.model.predict(state_tensor, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        # Train the model on a random sample from replay memory
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        current_q_values = self.model.predict(states, verbose=0)
        next_q_values_target = self.target_model.predict(next_states, verbose=0)

        targets = np.copy(current_q_values)
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values_target[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

    def decay_epsilon(self):
        # Reduce epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        print(f"Loading model weights from file: {name}")
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        print(f"Saving model weights to file: {name}")
        self.model.save_weights(name)