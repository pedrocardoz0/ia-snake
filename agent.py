import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        offset = 20
        point_l, point_r, point_u, point_d = [
            Point(head.x - offset, head.y),
            Point(head.x + offset, head.y),
            Point(head.x, head.y - offset),
            Point(head.x, head.y + offset),
        ]
        
        directions = [
            game.direction == Direction.LEFT,
            game.direction == Direction.RIGHT,
            game.direction == Direction.UP,
            game.direction == Direction.DOWN,
        ]

        state = [
            any(dir and game.is_collision(point) for dir, point in zip(directions, [point_r, point_l, point_u, point_d])),
            any(dir and game.is_collision(point) for dir, point in zip(directions[::-1], [point_r, point_l, point_u, point_d])),
            any(dir and game.is_collision(point) for dir, point in zip(directions[1:] + [directions[0]], [point_r, point_l, point_u, point_d])),
            *directions,
            game.food.x < game.head.x, game.food.x > game.head.x, game.food.y < game.head.y, game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        mini_sample = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(80 - self.n_games, 0)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            final_move[random.randint(0, 2)] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores, plot_mean_scores = [], []
    total_score, record = 0, 0
    agent, game = Agent(), SnakeGame()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} Score {score} Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()