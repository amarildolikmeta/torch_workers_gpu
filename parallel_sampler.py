from multiprocessing import Process, Queue, Event
import os
import time
import numpy as np
from utils import run_episodes


def traj_segment_function(model, env, weights, n_episodes, env_maker=None):
    '''
    Collects trajectories
    '''
    if n_episodes == 0:
        return []
    seed = np.random.randint(100000)
    returns = run_episodes(env=env, env_maker=env_maker, weights=weights, seed=seed, n_episodes=n_episodes, model=model)
    return returns


class Worker(Process):
    '''
    A worker is an independent process with its own environment and policy instantiated locally
    after being created. It ***must*** be runned before creating any torch session!
    '''

    def __init__(self, output, input, event, make_env, make_model, traj_segment_generator, seed, index):

        # Hide GPU
        child_env = os.environ.copy()
        child_env['CUDA_VISIBLE_DEVICES'] = ""
        super(Worker, self).__init__(kwargs={'env': child_env})

        self.output = output
        self.input = input
        self.make_env = make_env
        self.make_model = make_model
        self.traj_segment_generator = traj_segment_generator
        self.event = event
        self.seed = seed
        self.index = index

    def run(self):
        env = self.make_env()
        self.env = env
        env.reset()
        self.model = self.make_model()
        workerseed = self.seed + 10000 * np.random.randint(10000)
        env.seed(workerseed)
        while True:
            self.event.wait()
            self.event.clear()
            command, weights, n_episodes = self.input.get()
            if command == 'collect':
                samples = self.traj_segment_generator(self.model, env, weights, n_episodes)
                self.output.put((os.getpid(), samples))
            elif command == 'exit':
                print('Worker %s - Exiting...' % os.getpid())
                break


class ParallelSampler(object):

    def __init__(self, make_env, make_model, n_workers, seed=0):
        self.n_workers = n_workers
        print('Using %s CPUs' % self.n_workers)
        if seed is None:
            seed = time.time()

        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]

        fun = []
        for i in range(n_workers):
            f = lambda model, env, weights, n_episodes: traj_segment_function(model, env, weights, n_episodes,
                                                                              env_maker=make_env)
            fun.append(f)
        self.workers = [Worker(self.output_queue, self.input_queues[i], self.events[i],
                               make_env, make_model, fun[i], seed + i, i) for i in range(self.n_workers)]

        for w in self.workers:
            w.start()

    def collect(self, actor_weights, n_episodes):
        n_episodes_per_process = n_episodes // self.n_workers
        remainder = n_episodes % self.n_workers
        episodes = [n_episodes_per_process for _ in range(self.n_workers)]
        if remainder > 0:
            for i in range(remainder):
                episodes[i] += 1
        for i in range(self.n_workers):
            self.input_queues[i].put(('collect', actor_weights, episodes[i]))
        for e in self.events:
            e.set()
        sample_batches = []
        for i in range(self.n_workers):
            pid, samples = self.output_queue.get()
            sample_batches.extend(samples)
        return sample_batches

    def close(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None, None))
        for e in self.events:
            e.set()
        for w in self.workers:
            w.join()
