import pandas as pd
from gdrf2 import SparseMultinomialGDRF
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.nn as nn
import kernel
import pyro.distributions as dist
from utils import generate_data_2d_uniform
import os

use_cuda = True

def deprive(V:int = 100,
            W: int = 100,
            H: int = 100,
            l0: float = 0.01,
            sigma: float = 0.1,
            K: int = 5,
            NXu: int = 50,
            beta: float = 0.01,
            num_steps: int = 2000,
            lr: float = 0.001,
            num_particles: int = 1,
            seed: int = 1234567):
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    xs, ws = generate_data_2d_uniform(V=V, W=W, H=H, seed=seed, device='cuda' if use_cuda else 'cpu')

    pyro.set_rng_seed(seed)
    pyro.get_param_store().clear()


    lengthscale = nn.PyroParam(torch.tensor(float(l0), device='cuda' if use_cuda else 'cpu'), constraint=dist.constraints.positive)
    variance = nn.PyroParam(torch.tensor(sigma, device='cuda' if use_cuda else 'cpu'), constraint=dist.constraints.positive)
    k = kernel.RBF(1, lengthscale=lengthscale, variance=variance)
    if use_cuda:
        k = k.cuda()
    bounds = [(0.0, 1.0)]
    Xu = torch.rand(NXu, 1, device='cuda' if use_cuda else 'cpu')
    b = torch.tensor(beta, device='cuda' if use_cuda else 'cpu')
    gdrf = SparseMultinomialGDRF(b=b, k=K, v=V, s=k, Xu=Xu, n=xs.size(0), world=bounds, cuda=use_cuda, whiten=False)
    gdrf.train_model(x=xs, w=ws, num_steps=num_steps, lr=lr, num_particles=num_particles)
    ml_topics = gdrf.ml_topics(xs).reshape((W, H))
    random_words = gdrf.random_words(xs).reshape((W, H))

    folder = f"sdep_{K}_{l0}_{sigma}_{NXu}_{beta}_{num_steps}_{lr}_{num_particles}_{seed}"
    full_folder = os.path.join('..', 'data', folder)
    # if not os.path.exists(full_folder):
    #     os.mkdir(full_folder)
    plt.matshow(ml_topics)
    plt.show()
    plt.matshow(random_words)
    plt.show()
    wt_matrix = gdrf.word_topic_matrix
    plt.matshow(wt_matrix)
    plt.show()

if __name__ == "__main__":
    deprive(V = 100,
            W = 100,
            H = 100,
            l0 = 1.0,
            sigma = 10.0,
            K = 5,
            NXu = 50,
            beta = 1.0,
            num_steps = 500,
            lr = 0.001,
            num_particles = 1,
            seed = 1234567)

