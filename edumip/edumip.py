import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class EduMIP(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, max_episode_steps=500, render_mode=None):
        super().__init__()

        # Constants (simplified)
        self.gravity = 9.81  # m/s^2
        self.mass_body = 0.5  # kg
        self.mass_wheel = 0.05  # kg
        self.total_mass = self.mass_body + 2 * self.mass_wheel
        self.length = 0.1  # meters (center of mass height)
        self.inertia = 0.006  # kg*m^2 (body inertia around wheel axis)

        self.tau = 0.01  # seconds between state updates (100 Hz)

        self.max_torque = 2.0  # Nm
        self.max_theta = np.radians(30)  # max angle in radians

        # Action: 1D torque input [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: [x, x_dot, theta, theta_dot]
        high = np.array([np.finfo(np.float32).max]*4, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None
        self.steps = 0
        self.max_episode_steps = max_episode_steps

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400
        self.Noscale = 100  # 1 meter = 100 pixels
        self.cart_y = self.screen_height // 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        initial_x = 0.0
        initial_x_dot = 0.0
        initial_theta = np.random.uniform(low=-0.05, high=0.05)
        initial_theta_dot = np.random.uniform(low=-0.05, high=0.05)
        self.state = np.array([initial_x, initial_x_dot, initial_theta, initial_theta_dot], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        torque = np.clip(action[0], -1.0, 1.0) * self.max_torque

        x, x_dot, theta, theta_dot = self.state

        force = torque  # For simplicity treat torque directly as lateral force

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.mass_body * self.length * theta_dot**2 * sintheta) / (self.total_mass)

        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4/3 - (self.mass_body * costheta**2) / self.total_mass))
        xacc = temp - (self.mass_body * self.length * thetaacc * costheta) / self.total_mass

        x = x + x_dot * self.tau
        x_dot = x_dot + xacc * self.tau
        theta = theta + theta_dot * self.tau
        theta_dot = theta_dot + thetaacc * self.tau

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        self.steps += 1

        reward = 1.0
        # reward = np.clip(reward, 0.0, 1.0)

        terminated = abs(theta) > self.max_theta or abs(x) > 2.4
        truncated = self.steps >= self.max_episode_steps

        return self.state, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("EduMIP-v2")
            self.clock = pygame.time.Clock()

        wheel_radius = 20
        body_length = 100

        self.screen.fill((255, 255, 255))

        # Draw ground line
        ground_y = self.cart_y + 2 * wheel_radius + 10
        pygame.draw.line(self.screen, (0, 0, 0), (0, ground_y), (self.screen_width, ground_y), 2)

        x, _, theta, _ = self.state

        cart_x = int(self.screen_width // 2 + x * self.Noscale)
        cart_y = ground_y - wheel_radius

        # Draw the wheel
        pygame.draw.circle(self.screen, (0, 0, 0), (cart_x, cart_y), wheel_radius)

        # Draw the body (pole)
        x_tip = cart_x + body_length * np.sin(theta)
        y_tip = cart_y - wheel_radius - body_length * np.cos(theta)
        pygame.draw.line(self.screen, (0, 0, 255), (cart_x, cart_y - wheel_radius), (int(x_tip), int(y_tip)), 6)

        pygame.display.flip()
        self.clock.tick(50)
        self.clock.tick(50)