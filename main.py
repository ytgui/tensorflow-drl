import dqn


if __name__ == '__main__':
    agent = dqn.DQN()
    agent.train(episodes=2000)
    print('*******************************')
    agent.test(episodes=1000, max_step=500)