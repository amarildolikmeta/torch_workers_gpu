import numpy as np


def run_episodes(model, env, env_maker, weights, seed, n_episodes=1):

    # Random seed per process
    np.random.seed(seed)
    if env is None:
        env = env_maker()

    model.set_weights(weights)
    rets = []
    for i in range(n_episodes):
        s, _ = env.reset()
        done = False
        t = 0
        ret = 0
        while not done:
            P, _ = model.predict_one(s["nn_input"])
            action = np.random.choice(P.shape[0], p=P)
            s, reward, done, info = env.step(action)
            ret += reward
            t += 1
        rets.append(ret)
    return rets

