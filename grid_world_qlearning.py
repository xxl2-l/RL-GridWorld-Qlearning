import numpy as np
import pygame
import matplotlib.pyplot as plt
import sys

# 参数设置
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_SPACE = len(ACTIONS)
EPSILON = 0.9          # 初始探索率
EPSILON_DECAY = 0.995  # 每回合衰减
MIN_EPSILON = 0.01
ALPHA = 0.1            # 学习率
GAMMA = 0.9            # 折扣因子
EPISODES = 500         # 训练回合数

# 定义终点和陷阱
GOAL = (4, 0)          # 右上角
TRAPS = [(2,2), (1,3), (3,1)]  # 陷阱位置
WALLS = [(1,1), (3,3)]          # 墙壁（不可通过）

# 初始化 Q-table：大小为 (5,5,5) 因为状态是 (x,y)，每个状态有5个动作
Q = np.zeros((GRID_SIZE, GRID_SIZE, ACTION_SPACE))

def is_valid_state(x, y):
    """检查坐标是否在网格内且不是墙壁"""
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        return False
    if (x, y) in WALLS:
        return False
    return True

def get_next_state(x, y, action):
    """根据动作返回新坐标（如果无效则返回原坐标）"""
    if action == 0:   # UP
        new_x, new_y = x, y-1
    elif action == 1: # DOWN
        new_x, new_y = x, y+1
    elif action == 2: # LEFT
        new_x, new_y = x-1, y
    else:             # RIGHT
        new_x, new_y = x+1, y
    if is_valid_state(new_x, new_y):
        return new_x, new_y
    else:
        return x, y

def get_reward(x, y):
    """根据新位置返回奖励"""
    if (x, y) == GOAL:
        return 10
    elif (x, y) in TRAPS:
        return -5
    else:
        return -0.1   # 每步小惩罚

def train(render=False):
    epsilon = EPSILON
    episode_rewards = []
    for episode in range(EPISODES):
        x, y = 0, 4  # 起点
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 100:
            if render:
                # 处理退出事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                draw_grid(x, y)
                pygame.time.wait(100)  # 稍微放慢速度便于观察

            # ε-greedy 选择动作
            if np.random.rand() < epsilon:
                action = np.random.randint(ACTION_SPACE)  # 探索
            else:
                action = np.argmax(Q[x, y])               # 利用

            # 执行动作，得到新状态
            new_x, new_y = get_next_state(x, y, action)
            reward = get_reward(new_x, new_y)
            total_reward += reward

            # Q-learning 更新公式
            best_next_q = np.max(Q[new_x, new_y])
            Q[x, y, action] += ALPHA * (reward + GAMMA * best_next_q - Q[x, y, action])

            # 移动到新状态
            x, y = new_x, new_y
            steps += 1

            # 判断是否结束
            if (x, y) == GOAL or (x, y) in TRAPS:
                done = True

        # 衰减 epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
    return episode_rewards

# Pygame 初始化
pygame.init()
CELL_SIZE = 80
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Grid World Q-learning")
clock = pygame.time.Clock()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

def draw_grid(agent_x, agent_y):
    screen.fill(WHITE)
    # 绘制格子
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = pygame.Rect(col*CELL_SIZE, row*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            # 墙壁
            if (col, row) in WALLS:
                pygame.draw.rect(screen, BLACK, rect)
            # 陷阱
            elif (col, row) in TRAPS:
                pygame.draw.rect(screen, RED, rect)
            # 终点
            elif (col, row) == GOAL:
                pygame.draw.rect(screen, GREEN, rect)
            else:
                pygame.draw.rect(screen, GRAY, rect, 1)  # 空网格只画边框
            # 绘制边框
            pygame.draw.rect(screen, BLACK, rect, 1)

    # 绘制智能体（蓝色圆形）
    center_x = agent_x * CELL_SIZE + CELL_SIZE // 2
    center_y = agent_y * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, BLUE, (center_x, center_y), CELL_SIZE//3)
    pygame.display.flip()

if __name__ == "__main__":
    # 训练（不渲染，只输出文字）
    print("开始训练...")
    rewards = train(render=False)

    # 绘制训练曲线
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()

    # 训练结束后，用最佳策略演示一次
    print("训练完成，开始演示...")
    x, y = 0, 4
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        action = np.argmax(Q[x, y])  # 利用最佳策略
        new_x, new_y = get_next_state(x, y, action)
        # 不需要 reward 变量，可省略
        draw_grid(x, y)
        pygame.time.wait(500)  # 等待0.5秒
        x, y = new_x, new_y
        if (x, y) == GOAL or (x, y) in TRAPS:
            break
    pygame.time.wait(2000)
    pygame.quit()