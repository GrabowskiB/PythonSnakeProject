import os
import pygame
import sys
import random
import numpy as np
from dqn_agent import DQNAgent
import csv

# TensorFlow Configuration
try:
    import tensorflow as tf
    print("--- TensorFlow Configuration ---")
    print("TensorFlow Version:", tf.__version__)
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth enabled.")
    else:
        print("No GPU found. Training will run on CPU.")
except ImportError:
    print("ERROR: TensorFlow not installed.")
    sys.exit()

# Game Settings
screen_width, screen_height = 800, 600
block_size = 40
fps = 10
VISUALIZE_TRAINING = True

# Colors
white, red, green, blue = (255, 255, 255), (213, 50, 80), (0, 170, 0), (50, 153, 213)
background_color, grid_color = (40, 40, 40), (80, 80, 80)
snake_border_color, food_border_color = (0, 100, 0), (150, 0, 30)
head_color, q_value_highlight_color = (0, 230, 0), (255, 255, 0)

# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height)) if VISUALIZE_TRAINING else None
pygame.display.set_caption('Snake AI - DQN Visualization') if VISUALIZE_TRAINING else None
font_style = pygame.font.SysFont("bahnschrift", 30) if VISUALIZE_TRAINING else None
mesg_font_style = pygame.font.SysFont("arial", 40) if VISUALIZE_TRAINING else None
font_decision = pygame.font.SysFont("consolas", 18) if VISUALIZE_TRAINING else None
clock = pygame.time.Clock()

# Helper Functions
def draw_grid():
    if VISUALIZE_TRAINING and screen:
        for x in range(0, screen_width, block_size):
            pygame.draw.line(screen, grid_color, (x, 0), (x, screen_height))
        for y in range(0, screen_height, block_size):
            pygame.draw.line(screen, grid_color, (0, y), (screen_width, y))

def draw_snake(snake_block_size, snake_list):
    if VISUALIZE_TRAINING and screen:
        for x, y in snake_list[:-1]:
            pygame.draw.rect(screen, green, pygame.Rect(x, y, snake_block_size, snake_block_size))
        head_x, head_y = snake_list[-1]
        pygame.draw.rect(screen, head_color, pygame.Rect(head_x, head_y, snake_block_size, snake_block_size))

def draw_food(foodx, foody):
    if VISUALIZE_TRAINING and screen:
        pygame.draw.rect(screen, red, pygame.Rect(foodx, foody, block_size, block_size))

def generate_food_pos(snake_list):
    while True:
        foodx = random.randrange(0, screen_width // block_size) * block_size
        foody = random.randrange(0, screen_height // block_size) * block_size
        if all(segment != [foodx, foody] for segment in snake_list):
            return foodx, foody

def get_state(game_data):
    head, snake_body, food = game_data['head'], game_data['body'], game_data['food']
    current_dx, current_dy = game_data['dx'], game_data['dy']
    width, height, block = game_data['screen_width'], game_data['screen_height'], game_data['block_size']
    dir_vector = (current_dx // block, current_dy // block) if block else (1, 0)
    dir_straight, dir_right, dir_left = dir_vector, (dir_vector[1], -dir_vector[0]), (-dir_vector[1], dir_vector[0])
    point_s, point_r, point_l = (head[0] + dir_straight[0] * block, head[1] + dir_straight[1] * block), \
                                 (head[0] + dir_right[0] * block, head[1] + dir_right[1] * block), \
                                 (head[0] + dir_left[0] * block, head[1] + dir_left[1] * block)

    def is_collision(pt):
        return pt[0] >= width or pt[0] < 0 or pt[1] >= height or pt[1] < 0 or pt in snake_body[:-1]

    return np.array([
        int(is_collision(point_s)), int(is_collision(point_r)), int(is_collision(point_l)),
        int(dir_vector == (-1, 0)), int(dir_vector == (1, 0)), int(dir_vector == (0, -1)), int(dir_vector == (0, 1)),
        int(food[0] < head[0]), int(food[0] > head[0]), int(food[1] < head[1]), int(food[1] > head[1])
    ], dtype=int)

def map_action_to_direction(action_index, current_dx, current_dy, block_size):
    dir_vector = (current_dx // block_size, current_dy // block_size) if block_size else (1, 0)
    if action_index == 1:  # Left
        dir_vector = (-dir_vector[1], dir_vector[0])
    elif action_index == 2:  # Right
        dir_vector = (dir_vector[1], -dir_vector[0])
    return dir_vector[0] * block_size, dir_vector[1] * block_size

# AI Parameters
STATE_SIZE, ACTION_SIZE = 11, 3
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
TARGET_UPDATE_FREQ, LOAD_MODEL = 5, True
START_EPISODE_FROM_LOAD, MODEL_FILENAME = 800, "snake_dqn_model.weights.h5"
CSV_LOG_FILENAME = "snake_dqn_training_log.csv"

# Load Model
if LOAD_MODEL and os.path.exists(MODEL_FILENAME):
    agent.load(MODEL_FILENAME)
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * (agent.epsilon_decay ** START_EPISODE_FROM_LOAD))

# Game Loop
def game_loop(num_episodes=2000):
    global agent
    max_score = 0
    for episode in range(1, num_episodes + 1):
        lead_x, lead_y = (screen_width // 2 // block_size) * block_size, (screen_height // 2 // block_size) * block_size
        dx, dy, snake_list, length_of_snake = block_size, 0, [[lead_x, lead_y]], 1
        foodx, foody, score, steps, total_reward = generate_food_pos(snake_list), 0, 0, 0, 0
        game_over = False

        while not game_over:
            if VISUALIZE_TRAINING:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            state = get_state({'head': [lead_x, lead_y], 'body': snake_list, 'food': [foodx, foody],
                               'dx': dx, 'dy': dy, 'screen_width': screen_width, 'screen_height': screen_height,
                               'block_size': block_size})
            action = agent.act(state)
            dx, dy = map_action_to_direction(action, dx, dy, block_size)
            lead_x, lead_y = lead_x + dx, lead_y + dy
            steps += 1

            if lead_x >= screen_width or lead_x < 0 or lead_y >= screen_height or lead_y < 0 or [lead_x, lead_y] in snake_list[:-1]:
                game_over, reward = True, -100
            elif lead_x == foodx and lead_y == foody:
                reward, score, length_of_snake = 10, score + 1, length_of_snake + 1
                foodx, foody = generate_food_pos(snake_list)
                max_score = max(max_score, score)
            else:
                reward = 0

            snake_list.append([lead_x, lead_y])
            if len(snake_list) > length_of_snake:
                del snake_list[0]

            total_reward += reward
            next_state = get_state({'head': [lead_x, lead_y], 'body': snake_list, 'food': [foodx, foody],
                                    'dx': dx, 'dy': dy, 'screen_width': screen_width, 'screen_height': screen_height,
                                    'block_size': block_size})
            agent.remember(state, action, reward, next_state, game_over)
            agent.replay()

            if VISUALIZE_TRAINING:
                screen.fill(background_color)
                draw_grid()
                draw_food(foodx, foody)
                draw_snake(block_size, snake_list)
                pygame.display.flip()
                clock.tick(fps)

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
        agent.decay_epsilon()

        print(f"Episode: {episode}/{num_episodes}, Score: {score}, Max Score: {max_score}, Steps: {steps}, Total Reward: {total_reward}")

        with open(CSV_LOG_FILENAME, 'a', newline='') as f:
            csv.writer(f).writerow([episode, score, max_score, steps, total_reward, agent.epsilon])

        if episode % 50 == 0:
            agent.save(MODEL_FILENAME)

if __name__ == '__main__':
    game_loop(num_episodes=1000)