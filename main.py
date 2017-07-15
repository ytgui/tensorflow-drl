import dqn2013
import dqn2015


if __name__ == '__main__':
    agent = dqn2015.DQN()
    agent.train(episodes=2000, max_step=100)
    agent.test(episodes=1000, max_step=200)