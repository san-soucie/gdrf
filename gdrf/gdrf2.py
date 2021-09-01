import matplotlib

from collections import defaultdict

import torch
import pyro
import pyro.optim as optim
import pyro.nn as nn

import pyro.nn.module as module
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.infer as infer
import pyro.infer.autoguide as autoguide
from utils import generate_simple_data, generate_data_2d_circles, jittercholesky, dirichlet_param
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from tqdm import trange
import pyro.contrib.gp.kernels as kernel
import sys
import numpy as np

import wandb

class SparseMultinomialGDRF(gp.Parameterized):
    def __init__(self,
                 b: torch.Tensor,
                 k,
                 v,
                 Xu,
                 n,
                 world: list[tuple[float, float]],
                 m=lambda d: d.sum(dim=-1),
                 s: kernel.Kernel = kernel.Matern52(1),
                 jitter=1e-6,
                 f=lambda m: torch.softmax(m, -2),
                 cuda=True,
                 maxjitter: int = 3,
                 noise = None,
                 whiten=True):
        super().__init__()
        self.gpu = cuda
        self.device = 'cpu'
        if cuda:
            self.device = 'cuda'
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.mean_function = m
        self.kernel = s
        self.K = k
        self.V = v
        self.N = n
        beta = dirichlet_param(b, self.K, self.V)
        self.Xu = torch.nn.Parameter(Xu)
        self.Xu = nn.PyroParam(Xu, constraint=dist.constraints.stack([dist.constraints.interval(*c) for c in world], dim=1))
        self.beta = beta
        phi_map = torch.ones(self.beta.shape, dtype=self.beta.dtype) / self.V

        self.phi_map = nn.PyroParam(phi_map, constraint=dist.constraints.stack([dist.constraints.simplex for _ in range(self.K)], dim=0))
        self.f = f
        self.guide = None
        self.jitter = jitter
        self.maxjitter = maxjitter
        self.latent_shape = torch.Size((k,))
        self.M = Xu.size(0)
        self.D = Xu.size(1)
        u_loc = torch.zeros((self.K, self.M), dtype=Xu.dtype)
        self.u_loc = torch.nn.Parameter(u_loc)
        # self.u_loc = nn.PyroParam(u_loc)
        identity = dist.util.eye_like(Xu, self.M)
        u_scale_tril = identity.repeat((self.K, 1, 1))
        self.u_scale_tril = nn.PyroParam(u_scale_tril, dist.constraints.lower_cholesky)
        noise = torch.tensor(1.) if noise is None else noise
        self.noise = nn.PyroParam(noise, constraint=dist.constraints.positive)
        self.whiten = whiten

    @property
    def kernel_lengthscale(self):
        return pyro.param('kernel.lengthscale').detach().cpu().numpy()

    @property
    def kernel_variance(self):
        return pyro.param('kernel.variance').detach().cpu().numpy()


    @property
    def word_topic_matrix(self):
        return pyro.param('phi_map').detach().cpu().numpy()

    @property
    def word_topic_matrix_tensor(self):
        return pyro.param('phi_map')

    @property
    def inducing_points(self):
        return pyro.param('Xu').detach().cpu().numpy()

    @property
    def inducing_points_mean(self):
        return pyro.param('u_loc').detach().cpu().numpy()

    @property
    def inducing_points_scale_tril(self):
        return pyro.param('u_scale_tril').detach().cpu().numpy()

    def log_topic_probs(self, xs):
        posterior_kernel = self.kernel
        posterior_u_loc = self.inducing_points_mean
        posterior_u_scale_tril = self.inducing_points_scale_tril
        Luu = jittercholesky(
            posterior_kernel(torch.Tensor(self.inducing_points)).contiguous(),
            self.M,
            self.jitter,
            self.maxjitter
        )
        f_loc, _ = gp.util.conditional(
            xs,
            torch.Tensor(self.inducing_points),
            posterior_kernel,
            torch.Tensor(posterior_u_loc),
            torch.Tensor(posterior_u_scale_tril),
            Luu,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )
        return f_loc

    def topic_probs(self, xs):
        return self.f(self.log_topic_probs(xs)).detach().cpu().numpy().T

    def topic_probs_tensor(self, xs):
        return self.f(self.log_topic_probs(xs))

    def word_probs(self, xs):
        return self.topic_probs(xs) @ self.word_topic_matrix

    def word_probs_tensor(self, xs):
        return self.topic_probs_tensor(xs).T @ self.word_topic_matrix_tensor

    def ml_topics(self, xs):
        return np.argmax(self.topic_probs(xs), axis=1)

    def ml_words(self, xs):
        return np.argmax(self.word_probs(xs), axis=1)

    def random_words(self, xs, seed=None):
        rng = np.random.default_rng(seed=seed)
        probs = self.word_probs(xs)
        return np.array([rng.choice(a=self.V, p=probs[idx, :]) for idx in range(xs.shape[0])])



    @nn.pyro_method
    def model(self, x, w=None):

        Kuu = self.kernel(self.Xu).contiguous()
        Luu = jittercholesky(Kuu, self.M, self.jitter, self.maxjitter)
        u_scale_tril = dist.util.eye_like(self.Xu, self.M) if self.whiten else Luu
        zero_loc = self.Xu.new_zeros(self.u_loc.shape)

        f_loc, f_var = gp.util.conditional(
            x,
            self.Xu,
            self.kernel,
            self.u_loc,
            self.u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )

        f_loc = f_loc + self.mean_function(x)
        with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu') as idx:
            pyro.sample(
                "u",
                dist.MultivariateNormal(zero_loc, scale_tril=u_scale_tril).to_event(
                    zero_loc.dim() - 1
                ),
            )
            mu = pyro.sample("mu", dist.Normal(f_loc, f_var + self.noise).to_event(1))
            phi = pyro.sample("phi", dist.Dirichlet(self.beta))
        topic_probs = self.f(mu).transpose(-2, -1)
        probs = torch.matmul(topic_probs, phi)
        with pyro.plate("obs", device='cuda' if self.gpu else 'cpu'):
            w = pyro.sample("w", dist.Multinomial(probs=probs, validate_args=False), obs=w)
        return w

    def forward(self, x):
        return self.model(x)

    @nn.pyro_method
    def variational_distribution(self, x, w=None):
        kernel = self.kernel
        Xu = self.Xu
        u_loc = self.u_loc
        u_scale_tril = self.u_scale_tril
        Kuu = kernel(Xu).contiguous()
        Luu = jittercholesky(Kuu, self.M, self.jitter, self.maxjitter)
        f_loc, f_var = gp.util.conditional(
            x,
            Xu,
            kernel,
            u_loc,
            u_scale_tril,
            Luu,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )
        f_loc = f_loc + self.mean_function(x)
        phi_map = self.phi_map
        with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu') as idx:
            pyro.sample(
                "u",
                dist.MultivariateNormal(u_loc, scale_tril=u_scale_tril).to_event(
                    u_loc.dim() - 1
                ),
            )
            pyro.sample("mu", dist.Normal(f_loc, f_var).to_event(1))
            pyro.sample("phi", dist.Delta(phi_map).to_event(1))

    @nn.pyro_method
    def train_model(self, x, w, num_steps=100, lr=0.1, num_particles=10, disable_pbar=False,
                    early_stop=True, log=True):
        model = self.model
        guide = self.variational_distribution
        # guide = infer.autoguide.AutoDelta(model)

        optimizer = optim.AdamW({"lr": lr})
        objective = infer.TraceMeanField_ELBO(
            num_particles=num_particles,
            max_plate_nesting=1,
            vectorize_particles=True
        )

        svi = infer.SVI(model, guide, optimizer, objective)

        losses = []
        log_losses = []
        pbar = trange(num_steps, disable=disable_pbar)
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, n=name: gradient_norms[n].append(g.norm().item()))
        parameter_values = defaultdict(list)
        for idx in pbar:
            loss = svi.step(x, w)

            for name, _ in pyro.get_param_store().items():
                parameter_values[name] += [pyro.param(name).detach().cpu().numpy()]

            losses.append(loss)
            if log:
                p = self.perplexity_tensor(x, w).item()
                self.log_wandb(epoch=idx,
                               loss=loss,
                               perplexity=p,
                               noise=pyro.param('noise').detach().cpu().item(),
                               kernel_lengthscale=pyro.param('kernel.lengthscale').detach().cpu().item(),
                               kernel_variance=pyro.param('kernel.variance').detach().cpu().item(),
                               inducing_points=pyro.param('Xu').detach().cpu().numpy())
            if early_stop:
                log_losses.append(np.log(losses[-1]))
                running_log_loss_mean = np.mean(log_losses[-100:])
                recent_log_loss_resid = log_losses[-100:] - running_log_loss_mean
                loss_criterion = np.max(np.abs(recent_log_loss_resid)) / running_log_loss_mean
                if idx > 100 and loss_criterion < 1e-4:
                    print("Reached training convergence")
                    break

            pbar.set_description(f"Loss: {losses[-1]:10.10f}")
        return losses, gradient_norms, parameter_values

    def perplexity(self, x, w):
        return np.exp((w * np.log(self.word_probs(x))).sum() / -w.sum())

    def perplexity_tensor(self, x, w):
        return ((w * self.word_probs_tensor(x).log()).sum() / -w.sum()).exp()

    def log_wandb(self, **kwargs):
        wandb.log(kwargs)



def run_2d():
    K = 4
    V = 20
    W = 25
    H = 25
    R = 3
    N = 6
    NXu = 50
    l0 = 0.01
    eta = 0.001 / V
    constant_background = True
    permute = True

    beta = 0.0001
    nplot = 1
    num_steps = 2000
    lr = 0.005
    num_particles=100
    sigma = 0.01
    use_cuda = True

    topics_list = []
    words_list = []
    inferred_list = []

    for seed in range(nplot):
        pyro.set_rng_seed(seed)
        pyro.get_param_store().clear()
        lengthscale = torch.tensor(float(l0), device='cuda' if use_cuda else 'cpu')
        variance = torch.tensor(sigma, device='cuda' if use_cuda else 'cpu')
        k = kernel.RBF(2, lengthscale=lengthscale, variance=variance)
        if use_cuda:
            k = k.cuda()
        # kernel.variance = variance
        centers, topics, p_v_z, xs_raw, ws = generate_data_2d_circles(K=K, V=V, W=W, H=H, R=R, N=N, eta=eta, device='cuda' if use_cuda else 'cpu', permute=permute, constant_background=constant_background)
        xs = xs_raw.to(torch.get_default_dtype())
        xs[:, 0] /= W
        xs[:, 1] /= H
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        Xu = torch.rand(NXu, 2, device='cuda' if use_cuda else 'cpu')
        b = torch.tensor(beta)
        gdrf = SparseMultinomialGDRF(b=b, k=K, v=V, s=k, Xu=Xu, n=xs.size(0), world=bounds, cuda=use_cuda, whiten=False)
        losses, gs, ps = gdrf.train_model(x=xs, w=ws, num_steps=num_steps, lr=lr, num_particles=num_particles)
        epochs = list(range(len(losses)))

        plt.plot(epochs, losses)
        plt.yscale('log')
        plt.title("Training loss")
        plt.show()

        gdrf_container = gdrf
        #gdrf = gdrf.prior
        print(list(ps.keys()))
        print(list(gs.keys()))

        posterior_kernel = kernel.RBF(
            2,
            lengthscale=torch.Tensor(ps['kernel.lengthscale'][-1]),
            variance=torch.Tensor(ps['kernel.variance'][-1])
        )
        posterior_Xu = ps['Xu'][-1]
        posterior_u_loc = ps['u_loc'][-1]
        posterior_u_scale_tril = ps['u_scale_tril'][-1]
        Luu = jittercholesky(posterior_kernel(torch.Tensor(posterior_Xu)).contiguous(), gdrf.M, gdrf.jitter, gdrf.maxjitter)
        f_loc, _ = gp.util.conditional(
            xs,
            torch.Tensor(posterior_Xu),
            posterior_kernel,
            torch.Tensor(posterior_u_loc),
            torch.Tensor(posterior_u_scale_tril),
            Luu,
            full_cov=False,
            whiten=gdrf.whiten,
            jitter=gdrf.jitter,
        )
        phi_maps = [torch.tensor(x) for x in ps['phi_map']]
        phi_maps = [x.div(x.sum(dim=-1, keepdim=True)).detach().cpu().numpy() for x in phi_maps]
        mu_map = f_loc.reshape(K, W*H)
        # mu_map = torch.tensor(ps['AutoDelta.mu'][-1])
        phi_map = phi_maps[-1] # beta_map.div(beta_map.sum(dim=-1, keepdim=True))
        topic_probs = gdrf.f(mu_map.detach().cpu())

        ml_topics = torch.argmax(topic_probs, dim=0)
        ml_topic_map=torch.zeros(W, H, dtype=torch.int, device='cpu')


        for idx in range(len(ml_topics)):
            x, y = xs_raw[idx, :]
            ml_topic_map[x, y] = ml_topics[idx]
        Xu_raw = posterior_Xu
        Xu_raw[:, 0] *= W - 1
        Xu_raw[:, 1] *= H - 1

        logbounds = {'vmin': 0.01, 'vmax': 1.}

        plt.matshow(ml_topic_map.numpy())
        plt.scatter(Xu_raw[:, 0], Xu_raw[:, 1], marker='x', color='r')
        plt.title("Model maximum likelihood topics")
        plt.show()

        # frac = ws.T / torch.tensor(counts)
        plt.matshow(topics.detach().cpu().numpy())
        plt.title("Ground-truth topic distribution")
        plt.show()

        topic_coords, word_coords = np.mgrid[0:K:1, 0:V:1]
        def plot_probs(fig, ax, probs, title):

            pcm = ax.pcolor(topic_coords, word_coords, probs,
                            norm=colors.LogNorm(**logbounds), shading='auto')
            ax.set_title(title)
            ax.set_xticks(list(range(K)))
            ax.set_xticklabels([f'Topic {i}' for i in range(1, K+1)])
            ax.set_yticks(list(range(V)))
            ax.set_yticklabels([f'Word {i}' for i in range(1, V+1)])
            ax.tick_params(axis='x', labelrotation=45)
            fig.colorbar(pcm, ax=ax, extend='min')
            return pcm
        fig, ax = plt.subplots()
        plot_probs(fig, ax, phi_map, 'Model word distributions')
        plt.show()
        fig, ax = plt.subplots()
        plot_probs(fig, ax, p_v_z.detach().cpu().numpy(), 'Ground truth word distributions')
        plt.show()
        fig, ax = plt.subplots()
        plot_probs(fig, ax, phi_maps[0], "Initial phi")
        plt.show()
        fig, ax = plt.subplots()
        plot_probs(fig, ax, phi_maps[len(phi_maps) // 2], "Intermediate phi")
        plt.show()

        #
        # print(f"phi: {pyro.param('phi_q')}")
        # print(f"q: {q}")
        lengthscale = ps['kernel.lengthscale']
        plt.plot(epochs, lengthscale)
        plt.title("lengthscale")
        plt.show()

        random_words = gdrf.random_words(xs).reshape((W, H))
        plt.matshow(random_words)
        plt.title("Random words")
        plt.show()



        # print(list(gdrf.parameters()))
if __name__ == "__main__":
    run_2d()
