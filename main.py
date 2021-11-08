from bitFlipping import BitFlip
from actor import Actor
from parallel_sampler import ParallelSampler
from utils import run_episodes


def main():

    def env_maker():
        env = BitFlip(20, False, False)
        return env
    ## create main environment accessed in main process (no torch)
    env = env_maker()
    nn_state, _ = env.reset()
    obs_dim = nn_state["nn_input"].shape
    act_dim = env.n_actions
    lr = 0.0005

    def model_maker():
        # worker will not use GPU
        # Model is imported inside __init__
        return Actor(input_dim=obs_dim, output_dim=act_dim, lr=lr, use_gpu=False)
    seed = 42
    sampler_params = dict(
            n_workers=2,
            seed=4000,
        )
    ## Create worker processes with use_gpu=False
    sampler = ParallelSampler(make_env=env_maker, make_model=model_maker, **sampler_params)

    main_actor = Actor(input_dim=obs_dim, output_dim=act_dim, lr=lr, use_gpu=True)
    # main process collection
    it = 1
    while True:
        print("Iteration:", it)
        it += 1
        weights = main_actor.get_weights()
        rets = run_episodes(env=env, env_maker=env_maker(), weights=weights, seed=seed, n_episodes=2, model=main_actor)
        rets_parallel = sampler.collect(weights, 20)
    sampler.close()


if __name__ == '__main__': main()

