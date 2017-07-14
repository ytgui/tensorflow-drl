import dqn2013


if __name__ == '__main__':
    agent = dqn2013.DQN()
    agent.train(episodes=2000)
    agent.test(episodes=1000, max_step=500)