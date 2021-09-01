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
from utils import generate_simple_data, generate_data_2d_circles, jittercholesky
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from tqdm import trange
import kernel
import sys
import numpy as np
from likelihoods import MultiClass_Dirichlet


class GDRF(gp.Parameterized):
    def __init__(
        self,
        b,
        k,
        v,
        m=lambda d: d.sum(dim=-1),
        s: kernel.Kernel = kernel.Matern52(1),
        jitter=1e-6,
        f=lambda m: torch.softmax(m, 0),
        cuda=True,
        maxjitter: int = 3
    ):
        super().__init__()
        self.gpu = cuda
        if cuda:
            self.cuda()
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.mean_function = m
        self.kernel = s
        self.K = k
        self.V = v
        self.beta = torch.tensor(b)  # module.PyroParam(torch.tensor(b).cuda() if use_cuda else torch.tensor(b), dist.constraints.positive)
        self.f = f
        self.guide = None
        self.jitter = jitter
        self.maxjitter = maxjitter

    def model(self, x, w):
        N = x.size(0)
        Kff = self.kernel(x)
        Lff = jittercholesky(Kff, N, self.jitter, self.maxjitter)
        zero_loc = x.new_zeros(x.size(0))
        loc = zero_loc + self.mean_function(x)
        with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu'):
            mu = pyro.sample("mu", dist.MultivariateNormal(loc, scale_tril = Lff))
            phi = pyro.sample("phi", dist.Dirichlet(torch.Tensor([self.beta] * self.V)))
        with pyro.plate("obs", N, device='cuda' if self.gpu else 'cpu'):
            z = pyro.sample("z", dist.Categorical(self.f(mu).T))
            w = pyro.sample("w", dist.Categorical(pyro.ops.indexing.VIndex(phi)[:, z]), obs=w)
        return w

    def variational_distribution(self, x, w):
        N = x.size(0)
        mu_map = pyro.param('mu_map', torch.zeros(N, self.K))
        phi_map = pyro.param('phi_map', torch.ones(self.K, self.V) / self.V,
                             constraint=dist.constraints.simplex)
        with pyro.plate('topics', self.K, device='cuda' if self.gpu else 'cpu'):
            mu = pyro.sample("mu", dist.Delta(mu_map))
            phi = pyro.sample("phi", dist.Delta(phi_map))
        return w

    def train_model(self, x, w, num_steps=100, lr=0.1):
        # self.guide = autoguide.AutoGuideList(self.model)
        # self.guide.append(autoguide.AutoDelta(pyro.poutine.block(self.model, hide=["z", "w"])))
        # self.guide.append(autoguide.AutoDiscreteParallel(pyro.poutine.block(self.model, expose=["z", "w"])))
        self.guide = autoguide.AutoNormal(self.model, init_loc_fn=infer.autoguide.init_to_sample, init_scale=0.01)

        svi = infer.SVI(self.model, self.guide, optim.ClippedAdam({'lr': lr}), loss=infer.TraceMeanField_ELBO())

        losses = []
        pbar = trange(num_steps,)
        for _ in pbar:
            loss = svi.step(x, w)
            losses.append(loss)
            pbar.set_description(f"Loss: {losses[-1]:10.10f}")
        return losses

class MultinomialGDRF(GDRF):

    def model(self, x, w=None):
        N = x.size(0)
        # counts = w.sum(dim=1)
        Kff = self.kernel(x)
        Lff = jittercholesky(Kff, N, self.jitter, self.maxjitter)
        zero_loc = x.new_zeros(x.size(0))
        loc = zero_loc + self.mean_function(x)
        with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu'):
            mu = pyro.sample("mu", dist.MultivariateNormal(loc, scale_tril=Lff))
            phi = pyro.sample("phi", dist.Dirichlet(torch.Tensor([self.beta] * self.V)))
        with pyro.plate("obs", N, device='cuda' if self.gpu else 'cpu'):
            w = pyro.sample("w", dist.Multinomial(probs=self.f(mu).T @ phi, validate_args=False), obs=w)
        return mu, phi, w

    def variational_distribution(self, x, w=None):
        N = x.size(0)
        M_q = pyro.param("m_q", (torch.zeros(self.K, x.size(0))))
        l_q = pyro.param("l_q", torch.tensor(0.2), constraint=dist.constraints.positive)
        s_q = pyro.param("s_q", torch.tensor(0.01), constraint=dist.constraints.positive)
        kernel = self.kernel
        kernel.lengthscale = l_q
        kernel.variance = s_q
        Kff = kernel(x)
        Lff = jittercholesky(Kff, N, self.jitter, self.maxjitter)
        phi_q = pyro.param("phi_q", torch.ones((self.K, self.V)) / self.V, constraint=dist.constraints.stack([dist.constraints.simplex for _ in range(self.K)], dim=0))
        with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu'):
            mu = pyro.sample("mu", dist.MultivariateNormal(M_q, scale_tril=Lff))
            phi = pyro.sample("phi", dist.Delta(phi_q).to_event(1))
        return mu, phi

    def train_model(self, x, w, num_steps=100, lr=0.1):

        svi = infer.SVI(self.model, self.variational_distribution, optim.ClippedAdam({'lr': lr, 'betas': (0.95, 0.999), 'lrd': (lr / 100) ** (1 / num_steps)}), loss=infer.TraceGraph_ELBO(num_particles=1, max_plate_nesting=1))

        losses = []
        pbar = trange(num_steps,)
        for _ in pbar:
            loss = svi.step(x, w)
            losses.append(loss)
            pbar.set_description(f"Loss: {losses[-1]:10.10f}")
        return losses

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
            self.cuda()
            self.device = 'cuda'
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.mean_function = m
        self.kernel = s
        self.K = k
        self.V = v
        self.N = n
        if len(b.shape) == 0:
            beta = torch.ones(self.K, self.V) * b
        elif len(b.shape) == 1:
            if b.shape[0] == self.K:
                beta = b.repeat(self.V, 1).T
            elif b.shape[0] == self.V:
                beta = b.repeat(self.K, 1)
            else:
                raise ValueError("parameter b must have length K or V if 1D")
        elif len(b.shape) == 2:
            assert b.shape == torch.Size((self.K, self.V)), "b should be KxV if 2D"
            beta = b
        else:
            raise ValueError("invalid b parameter- you passed %s", b)
        self.Xu = torch.nn.Parameter(Xu)
        self.Xu = nn.PyroParam(Xu, constraint=dist.constraints.stack([dist.constraints.interval(*c) for c in world], dim=1))
        self.beta = beta
        self.phi_map = torch.nn.Parameter(beta.div(beta.sum(dim=-1, keepdim=True)))

        self.phi_map = nn.PyroParam(self.phi_map, constraint=dist.constraints.stack([dist.constraints.simplex for _ in range(self.K)], dim=0))
        self.M_q = torch.nn.Parameter(torch.zeros(self.K, self.N, device=self.device))
        self.M_q = nn.PyroParam(torch.zeros(self.K, self.N, device=self.device))
        self.f = f
        self.guide = None
        self.jitter = jitter
        self.maxjitter = maxjitter
        self.latent_shape = torch.Size((k,))
        self.M = self.Xu.size(0)
        #u_loc = torch.randn((self.K, self.M), dtype=self.Xu.dtype)
        #self.u_loc = torch.nn.Parameter(u_loc)
        identity = dist.util.eye_like(self.Xu, self.M)
        # u_scale_tril = identity.repeat((self.K, 1, 1))
        # self.u_scale_tril = nn.PyroParam(u_scale_tril, dist.constraints.lower_cholesky)
        noise = torch.tensor(1.) if noise is None else noise
        self.noise = nn.PyroParam(noise, constraint=dist.constraints.positive)
        self.whiten = whiten


    def model(self, x, w=None):
        self.set_mode("model")

        N = x.size(0)
        M = self.Xu.size(0)

        Kuu = self.kernel(self.Xu).contiguous()
        Luu = jittercholesky(Kuu, M, self.jitter, self.maxjitter)
        Kuf = self.kernel(self.Xu, x)
        W = Kuf.triangular_solve(Luu, upper=False)[0].t()
        D = self.noise.expand(N)

        zero_loc = x.new_zeros(x.size(0))
        loc = zero_loc + self.mean_function(x)
        with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu'):
            mu = pyro.sample("mu", dist.LowRankMultivariateNormal(loc, W, D))
            phi = pyro.sample("phi", dist.Dirichlet(self.beta))
        probs = torch.matmul(self.f(mu).transpose(-2, -1), phi)
        with pyro.plate("obs", N, device='cuda' if self.gpu else 'cpu') as idx:
            w = pyro.sample("w", dist.Multinomial(probs=probs[..., idx, :], validate_args=False), obs=w[..., idx, :])
        return mu, phi, w

    def variational_distribution(self, x, w=None):
        self.set_mode("guide")
        self._load_pyro_samples()
        N = x.size(0)
        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()
        Luu = jittercholesky(Kuu, M, self.jitter, self.maxjitter)
        Kuf = self.kernel(self.Xu, x)
        W = Kuf.triangular_solve(Luu, upper=False)[0].t()
        D = self.noise.expand(N)
        W_map = pyro.sample("W_map", dist.Delta(W).to_event(2), infer={'is_auxiliary': True})
        D_map = pyro.sample("D_map", dist.Delta(D).to_event(1), infer={'is_auxiliary': True})
        M_map = pyro.sample("M_map", dist.Delta(self.M_q).to_event(2), infer={'is_auxiliary': True})
        with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu'):

            mu = pyro.sample("mu", dist.LowRankMultivariateNormal(M_map, W_map, D_map))
            phi = pyro.sample("phi", dist.Delta(self.phi_map).to_event(1))
        return mu, phi

    # def model(self, x, w=None):
    #     self.set_mode("model")
    #     N = x.size(0)
    #     Kuu = self.kernel(self.Xu).contiguous()
    #     Luu = jittercholesky(Kuu, self.M, self.jitter, self.maxjitter)
    #     zero_loc = self.Xu.new_zeros(self.u_loc.shape)
    #     identity = dist.util.eye_like(self.Xu, self.M)
    #     scale_tril = identity if self.whiten else Luu
    #     with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu'):
    #         pyro.sample(
    #             self._pyro_get_fullname("u"),
    #             dist.MultivariateNormal(zero_loc, scale_tril=scale_tril).to_event(zero_loc.dim() - 1),
    #         )
    #         f_loc, f_var = gp.util.conditional(
    #             x,
    #             self.Xu,
    #             self.kernel,
    #             self.u_loc,
    #             self.u_scale_tril,
    #             Luu,
    #             full_cov=False,
    #             whiten=self.whiten,
    #             jitter=self.jitter,
    #         )
    #         f_loc += self.mean_function(x)
    #         #f_cov.view(-1)[:: N + 1] += self.noise
    #         mu = pyro.sample("mu", dist.Normal(f_loc, f_var).to_event(1))
    #         phi = pyro.sample("phi", dist.Dirichlet(torch.Tensor([self.beta] * self.V)))
    #     probs = torch.matmul(self.f(mu).transpose(-2, -1), phi)
    #     with pyro.plate("obs", N, device='cuda' if self.gpu else 'cpu') as idx:
    #         w = pyro.sample("w", dist.Multinomial(probs=probs[..., idx, :], validate_args=False), obs=w[..., idx, :])
    #     return mu, phi, w

    # def model(self, x, w=None):
    #     self.set_mode("model")
    #     N = x.size(0)
    #     Kuu = self.kernel(self.Xu).contiguous()
    #     Luu = jittercholesky(Kuu, self.M, self.jitter, self.maxjitter)
    #     zero_loc = self.Xu.new_zeros(self.u_loc.shape)
    #     identity = dist.util.eye_like(self.Xu, self.M)
    #     scale_tril = identity if self.whiten else Luu
    #     f_loc = x.new_zeros(N) + self.mean_function(x)
    #     Kff = self.kernel(x).contiguous()
    #     f_scale_tril = jittercholesky(Kff, N, self.jitter, self.maxjitter)
    #     with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu'):
    #         pyro.sample(
    #             self._pyro_get_fullname("u"),
    #             dist.MultivariateNormal(zero_loc, scale_tril=scale_tril).to_event(zero_loc.dim() - 1),
    #         )
    #         mu = pyro.sample("mu", dist.MultivariateNormal(f_loc, scale_tril=f_scale_tril))
    #         phi = pyro.sample("phi", dist.Dirichlet(torch.Tensor([self.beta] * self.V)))
    #     probs = torch.matmul(self.f(mu).transpose(-2, -1), phi)
    #     with pyro.plate("obs", N, device='cuda' if self.gpu else 'cpu') as idx:
    #         w = pyro.sample("w", dist.Multinomial(probs=probs[..., idx, :], validate_args=False), obs=w[..., idx, :])
    #     return mu, phi, w
    #
    # def variational_distribution(self, x, w=None):
    #     self.set_mode("guide")
    #     self._load_pyro_samples()
    #     N = x.size(0)
    #     Kuu = self.kernel(self.Xu).contiguous()
    #     Luu = jittercholesky(Kuu, self.M, self.jitter, self.maxjitter)
    #     beta_q = pyro.param("beta_q", torch.rand((self.K, self.V)), constraint=dist.constraints.positive)
    #     with pyro.plate("topics", self.K, device='cuda' if self.gpu else 'cpu'):
    #         pyro.sample(
    #             self._pyro_get_fullname("u"),
    #             dist.MultivariateNormal(self.u_loc, scale_tril=self.u_scale_tril).to_event(self.u_loc.dim() - 1),
    #         )
    #         f_loc, f_var = gp.util.conditional(
    #             x,
    #             self.Xu,
    #             self.kernel,
    #             self.u_loc,
    #             self.u_scale_tril,
    #             Luu,
    #             full_cov=False,
    #             whiten=self.whiten,
    #             jitter=self.jitter,
    #         )
    #         f_loc += self.mean_function(x)
    #         #f_cov.view(-1)[:: N + 1] += self.noise
    #         mu = pyro.sample("mu", dist.Normal(f_loc, f_var).to_event(1))
    #         phi = pyro.sample("phi", dist.Dirichlet(beta_q))
    #     return mu, phi, None

    def train_model(self, x, w, num_steps=100, lr=0.1, num_particles=10):
        model = self.model
        guide = self.variational_distribution
        optimizer = optim.ClippedAdam({'lr': lr})
        objective = infer.TraceMeanField_ELBO(num_particles=num_particles, max_plate_nesting=1, vectorize_particles=True)
        svi = infer.SVI(model, guide, optimizer, loss=objective)

        losses = []
        pbar = trange(num_steps)
        for _ in pbar:
            loss = svi.step(x, w)
            losses.append(loss)
            pbar.set_description(f"Loss: {losses[-1]:10.10f}")
        return losses

    def train_model_store_values(self, x, w, num_steps=100, lr=0.1, num_particles=10):
        model = self.model
        guide = self.variational_distribution
        optimizer = optim.ClippedAdam({'lr': lr})
        objective = infer.Trace_ELBO(num_particles=num_particles, max_plate_nesting=2, vectorize_particles=True)
        svi = infer.SVI(model, guide, optimizer, loss=objective)

        losses = []
        pbar = trange(num_steps)
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, n=name: gradient_norms[n].append(g.norm().item()))
        parameter_values = defaultdict(list)
        for _ in pbar:
            loss = svi.step(x, w)
            for name, value in pyro.get_param_store().named_parameters():
                parameter_values[name] += [value.detach().cpu().numpy()]
            parameter_values["beta"] += [self.beta.detach().cpu().numpy()]
            parameter_values["Xu"] += [self.Xu.detach().cpu().numpy()]

            losses.append(loss)
            pbar.set_description(f"Loss: {losses[-1]:10.10f}")
        return losses, gradient_norms, parameter_values

    def train_model_store_values2(self, x, w, num_steps=100, lr=0.1, num_particles=10):
        model = self.model
        guide = self.variational_distribution

        optimizer = torch.optim.Adamax(
            self.parameters(),
            lr=lr,
        )
        objective = infer.Trace_ELBO(
            num_particles=num_particles,
            max_plate_nesting=1,
            vectorize_particles=True
        ).differentiable_loss

        def closure():
            optimizer.zero_grad()
            loss = objective(model, guide, x, w)
            infer.util.torch_backward(loss, retain_graph=None)
            return loss

        losses = []
        log_losses = []
        pbar = trange(num_steps)
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, n=name: gradient_norms[n].append(g.norm().item()))
        parameter_values = defaultdict(list)
        for idx in pbar:
            loss = optimizer.step(closure)
            for name, value in pyro.get_param_store().named_parameters():
                parameter_values[name] += [value.detach().cpu().numpy()]
            parameter_values["beta"] += [self.beta.detach().cpu().numpy()]
            parameter_values["Xu"] += [self.Xu.detach().cpu().numpy()]

            losses.append(loss.detach().cpu().numpy())
            log_losses.append(np.log(losses[-1]))
            running_log_loss_mean = np.mean(log_losses[-100:])
            recent_log_loss_resid = log_losses[-100:] - running_log_loss_mean
            loss_criterion = np.max(np.abs(recent_log_loss_resid)) / running_log_loss_mean
            if idx > 100 and loss_criterion < 1e-4:
                print("Reached training convergence")
                break

            pbar.set_description(f"Loss: {losses[-1]:10.10f}")
        return losses, gradient_norms, parameter_values

def test_simple():
    K=2
    V=2
    N=100
    poisson_rate = 1000.
    radius=0.5
    beta=0.1
    nplot=2
    num_steps=600
    lr=0.01
    sigma=0.5
    use_cuda = True

    topics_list = []
    words_list = []
    inferred_list = []

    for seed in range(nplot):
        pyro.get_param_store().clear()
        lengthscale = torch.tensor(float(radius))
        variance = torch.tensor(sigma)
        if use_cuda:
            lengthscale = lengthscale.cuda()
            variance = variance.cuda()
        k = kernel.Matern52(1, lengthscale=lengthscale, variance=variance)
        # kernel.variance = variance
        p, q, counts, ws = generate_simple_data(N=N, poisson_rate=poisson_rate, seed=seed, device='cuda' if use_cuda else 'cpu')
        xs = torch.arange(0, 1., 1/N, device='cuda' if use_cuda else 'cpu')
        gdrf = MultinomialGDRF(b=beta, k=K, v=V, s=k, cuda=use_cuda)
        losses = gdrf.train_model(x=xs, w=ws, num_steps=num_steps, lr=lr)
        plt.plot(list(range(num_steps)), losses)
        plt.title("Training loss")
        plt.show()

        mu_map = pyro.param('m_q')
        phi_map = pyro.param('phi_q')
        topic_probs = torch.softmax(mu_map.detach().cpu(), 0).numpy()
        wt_matrix = phi_map.detach().cpu().T.numpy()
        plt.matshow(wt_matrix @ topic_probs)
        plt.title("Model word probabilities")
        plt.show()

        frac = ws.T / torch.tensor(counts)
        plt.matshow(frac.detach().cpu().numpy())
        plt.title("Observed word distributions")
        plt.show()
        #
        # print(f"phi: {pyro.param('phi_q')}")
        # print(f"q: {q}")

def test_2d():
    K = 3
    V = 12
    W = 25
    H = 25
    R = 3
    N = 5
    NXu = 50
    l0 = 0.1
    eta = 0.2 / V
    constant_background=False
    permute=True

    beta = 0.001
    nplot = 1
    num_steps = 1000
    lr = 0.1
    num_particles=30
    sigma = 0.1
    use_cuda = True

    topics_list = []
    words_list = []
    inferred_list = []

    for seed in range(nplot):
        pyro.get_param_store().clear()
        lengthscale = nn.PyroParam(torch.tensor(float(l0), device='cuda' if use_cuda else 'cpu'), constraint=dist.constraints.positive)
        variance = nn.PyroParam(torch.tensor(sigma, device='cuda' if use_cuda else 'cpu'), constraint=dist.constraints.positive)
        k = kernel.RBF(2, lengthscale=lengthscale, variance=variance)
        if use_cuda:
            k = k.cuda()
        # kernel.variance = variance
        centers, topics, p_v_z, xs_raw, ws = generate_data_2d_circles(
            K=K,
            V=V,
            W=W,
            H=H,
            R=R,
            N=N,
            eta=eta,
            device='cuda' if use_cuda else 'cpu',
            permute=permute,
            constant_background=constant_background
        )
        xs = xs_raw.to(torch.get_default_dtype())
        xs[:, 0] /= W
        xs[:, 1] /= H
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        Xu = torch.rand(NXu, 2, device='cuda' if use_cuda else 'cpu')
        b = torch.tensor(beta)
        gdrf = SparseMultinomialGDRF(b=b, k=K, v=V, s=k, Xu=Xu, n=xs.size(0), world=bounds, cuda=use_cuda, whiten=True)
        losses, gs, ps = gdrf.train_model_store_values2(x=xs, w=ws, num_steps=num_steps, lr=lr, num_particles=num_particles)
        epochs = list(range(len(losses)))

        plt.plot(epochs, losses)
        plt.yscale('log')
        plt.title("Training loss")
        plt.show()

        mu_map = gdrf.M_q
        beta_map = gdrf.beta
        phi_map = gdrf.phi_map  # beta_map.div(beta_map.sum(dim=-1, keepdim=True))
        topic_probs = torch.softmax(mu_map.detach().cpu(), 0)

        ml_topics = torch.argmax(topic_probs, dim=0)
        ml_topic_map=torch.zeros(W, H, dtype=torch.int, device='cpu')


        for idx in range(len(ml_topics)):
            x, y = xs_raw[idx, :]
            ml_topic_map[x, y] = ml_topics[idx]
        Xu_raw = gdrf.Xu.to(torch.get_default_dtype())
        Xu_raw[:, 0] *= W
        Xu_raw[:, 1] *= H
        Xu_raw = Xu_raw.detach().cpu().numpy()

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
        plot_probs(fig, ax, phi_map.detach().cpu().numpy(), 'Model word distributions')
        plt.show()
        fig, ax = plt.subplots()
        plot_probs(fig, ax, p_v_z.detach().cpu().numpy(), 'Ground truth word distributions')
        plt.show()

        #
        # print(f"phi: {pyro.param('phi_q')}")
        # print(f"q: {q}")
        lengthscale = ps['kernel.lengthscale']
        plt.plot(epochs, lengthscale)
        plt.title("lengthscale")
        plt.show()


        # print(list(gdrf.parameters()))

test_2d()

class VariationalGDRF(pyro.contrib.gp.models.VariationalGP):

    @nn.module.pyro_method
    def guide(self):
        super().guide()
        f_var = self.f_scale_tril.pow(2).sum(dim=-1)
        if self.y is not None:
            self.likelihood(self.f_loc, f_var, self.y)


# # K=5
# # V=100
# # n_obj=8
# # width=75
# # height=32
# # radius=8
# # eta=0.1
# # beta=0.1
# # nplot=2
# # num_steps=1000
# # lr=0.1
# # sigma=2
# # use_cuda = True
# #
# # topics_list = []
# # words_list = []
# # inferred_list = []
# #
# # for seed in range(nplot):
# #     pyro.get_param_store().clear()
# #     lengthscale = torch.tensor(float(radius))
# #     variance = torch.tensor(sigma)
# #     if use_cuda:
# #         lengthscale = lengthscale.cuda()
# #         variance = variance.cuda()
# #     kernel = gp.kernels.Matern52(0, lengthscale=lengthscale)
# #     kernel.variance = variance
# #
# #     centers, topics, p_v_z, words = generate_data(K=K, V=V, N=n_obj, W=width, H=height, R=radius, eta=eta, seed=seed)
# #     xs = []
# #     ws = []
# #
# #     for x in range(width):
# #         for y in range(height):
# #             xs.append(x)
# #             ws.append(words[x, y])
# #     xs = torch.Tensor(xs)
# #     ws = torch.Tensor(ws)
# #     if use_cuda:
# #         xs = xs.cuda()
# #         ws = ws.cuda()
# #     gdrf = GDRF(b=beta, k=K, v=V, s=kernel, cuda=use_cuda)
# #     losses = gdrf.train_model(x=xs, w=ws, num_steps=num_steps, lr=lr)
# #
# #     plt.plot(list(range(num_steps)), losses)
# #     plt.title("Training loss")
# #     plt.show()
#
#     # serving_model = infer.infer_discrete(gdrf.model, first_available_dim=-2, temperature=0)
#     # mu, phi, z, _ = serving_model(xs, ws)
#     # inferred_topics = torch.zeros((width, height), dtype=torch.int)
#     # i = 0
#     # for x in range(width):
#     #     for y in range(height):
#     #         inferred_topics[x, y] = z[i].cpu().detach()
#     #         i += 1
#     # topics_list.append(topics.cpu().detach())
#     # words_list.append(words.cpu().detach())
#     # inferred_list.append(inferred_topics.cpu().detach())
#     del gdrf

# fig, axs = plt.subplots(nplot, 3, sharex=True, sharey=True)
# fig.set_size_inches(15.75, 15.75)
# for i in range(nplot):
#     _, topics, _, words = generate_data(seed=i)
#     axs[i, 0].matshow(topics_list[i].numpy().T, cmap=plt.get_cmap('prism'))
#     axs[i, 1].matshow(words_list[i].numpy().T, cmap=plt.get_cmap('prism'))
#     axs[i, 2].matshow(inferred_list[i].numpy().T, cmap=plt.get_cmap('tab10'))
#     axs[i, 0].xaxis.tick_bottom()
#     axs[i, 1].xaxis.tick_bottom()
#     axs[i, 2].xaxis.tick_bottom()
# plt.show()
