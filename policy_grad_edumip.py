import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import os
import continuous_cartpole  # Make sure this line registers 'ContinuousCartPole-v0'
import edumip

import matplotlib.pyplot as plt


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "preview"], required=True)
args = parser.parse_args()

# Create environment
env = gym.make("EduMIP-v2", render_mode="human" if args.mode == "preview" else None)
env = gym.wrappers.TimeLimit(env, max_episode_steps=500)

# Hyperparameters
gamma = 0.99
learning_rate = 0.0002
max_episodes = 9000
model_save_path = "model_policy_edumip.keras"

# Actor Network
# def create_actor():
#     inputs = keras.Input(shape=(env.observation_space.shape[0],))
#     x = layers.Dense(64, activation="relu")(inputs)
#     x = layers.Dense(64, activation="relu")(x)
#     mu = layers.Dense(1, activation="tanh")(x)  # Output mean between -1 and 1
#     sigma = layers.Dense(1, activation="softplus")(x)  # Output positive std deviation
#     return keras.Model(inputs=inputs, outputs=[mu, sigma])

actor = keras.models.load_model("model_policy_cartpole.keras") # Bootstrap off of the continuous cartpole.
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Function to sample an action
def sample_action(mu, sigma, episode):
    if episode > 1000:
        sigma = tf.maximum(0.01, sigma * 0.99)  # Decay sigma by 1% per episode, floor at 0.01
    action = tf.random.normal(shape=[1], mean=mu, stddev=sigma)
    return tf.clip_by_value(action, -1.0, 1.0)

# Function to compute returns (discounted rewards)
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    # returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
    return returns

if args.mode == "train":
    all_rewards = []
    all_loss = []

    # Set up live plotting
    plt.ion()
    fig, ax = plt.subplots()
    rewards_plot, = ax.plot([], [])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Progress')

    try:
        for episode in range(1, max_episodes + 1):
            state, _ = env.reset()
            done = False
            episode_states = []
            episode_actions = []
            episode_rewards = []

            while not done:
                state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
                mu, sigma = actor(state_tensor)
                action = sample_action(mu, sigma, episode)

                next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
                done = terminated or truncated

                episode_states.append(state)
                episode_actions.append(action.numpy()[0])
                episode_rewards.append(reward)

                state = next_state

            returns = compute_returns(episode_rewards, gamma)

            with tf.GradientTape() as tape:
                mu_sigma = actor(tf.convert_to_tensor(np.vstack(episode_states), dtype=tf.float32))
                mus, sigmas = mu_sigma[0], mu_sigma[1]

                sigmas = tf.clip_by_value(sigmas, 1e-4, 1.0)  # Avoid tiny sigmas that blow up

                actions_tensor = tf.convert_to_tensor(episode_actions, dtype=tf.float32)

                # Manual log probability calculation
                log_probs = -0.5 * tf.math.log(2.0 * np.pi * tf.square(sigmas)) \
                            - 0.5 * tf.square(actions_tensor - mus) / tf.square(sigmas)
                log_probs = tf.squeeze(log_probs, axis=-1)

                loss = -tf.reduce_mean(log_probs * returns)

            grads = tape.gradient(loss, actor.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor.trainable_variables))

            all_rewards.append(np.sum(episode_rewards))
            all_loss.append(loss.numpy())

            if episode % 10 == 0:
                rewards_plot.set_xdata(np.arange(len(all_rewards)))
                rewards_plot.set_ydata(all_rewards)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)

            if episode % 100 == 0:
                print(f"Episode {episode}: Average Reward = {np.mean(all_rewards[-100:]):.2f}, Loss = {np.mean(all_loss[-100:]):.4f}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        
    # Save the trained model
    actor.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    env.close()

elif args.mode == "preview":
    env = gym.make("EduMIP-v2", render_mode="human")
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model file {model_save_path} not found. Train the model first.")

    actor = keras.models.load_model(model_save_path)
    print(f"Model loaded from {model_save_path}")

    test_episodes = 5
    for episode in range(test_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
            mu, sigma = actor(state_tensor)
            action = tf.clip_by_value(mu, -1.0, 1.0)
            next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
            done = terminated or truncated
            state = next_state
            total_reward += reward
            env.render()
        print(f"Test Episode {episode+1}: Total Reward = {total_reward}")


