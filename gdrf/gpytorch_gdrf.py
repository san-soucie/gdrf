import torch
import pyro
import pyro.distributions as dist
import gpytorch
from gpytorch.variational import DeltaVariationalDistribution, IndependentMultitaskVariationalStrategy, VariationalStrategy
from utils import dirichlet_param, generate_data_2d_circles
import pyro.optim as optim
import pyro.infer as infer
from tqdm import tqdm, trange
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cbook as cbook


class SparseMultinomialGDRF(gpytorch.models.ApproximateGP):
    def __init__(self,
                 b: torch.Tensor,
                 k,
                 v,
                 Xu,
                 f=lambda m: torch.softmax(m, -2),
                 cuda=True,
                 name_prefix="gdrf"):
        self.K = k
        self.V = v
        self.f = f
        self.name_prefix = name_prefix
        self.cuda = cuda
        self.device = 'cuda' if self.cuda else 'cpu'
        self.beta = dirichlet_param(b, self.K, self.V)
        inducing_points = Xu
        num_inducing_points = inducing_points.size(-2)
        batch_shape = torch.Size([self.K])
        variational_distribution = DeltaVariationalDistribution(num_inducing_points=num_inducing_points, batch_shape=batch_shape)
        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=self.K,
            task_dim=-2
        )
        super().__init__(variational_strategy=variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def guide(self, x, w=None):
        phi_map = pyro.param(self.name_prefix + ".phi_map", self.beta.div(self.beta.sum(dim=-1, keepdim=True)), event_dim=1)
        function_distribution = self.pyro_guide(x, name_prefix=self.name_prefix)
        mu = pyro.sample(self.name_prefix + ".mu", function_distribution.to_event(1))
        phi = pyro.sample(self.name_prefix + ".phi", dist.Delta(phi_map))

    def model(self, x, w=None):
        pyro.module(self.name_prefix + ".gp", self)
        function_distribution = self.pyro_model(x, name_prefix=self.name_prefix)
        mu = pyro.sample(self.name_prefix + ".mu", function_distribution)
        phi = pyro.sample(self.name_prefix + ".phi", dist.Dirichlet(self.beta).to_event(1))
        probs = torch.matmul(self.f(mu).transpose(-2, -1), phi)
        with pyro.plate("obs", device='cuda' if self.gpu else 'cpu'):
            obs = pyro.sample(self.name_prefix + ".w", dist.Multinomial(probs=probs, validate_args=False), obs=w)
        return obs

    def train_model(self, x, w, num_steps=100, lr=0.1, num_particles=10):
        model = self.model
        guide = pyro.infer.autoguide.AutoDelta(self.model)

        optimizer = optim.SGD({"lr": lr})
        objective = infer.Trace_ELBO(
            num_particles=num_particles,
            max_plate_nesting=1,
            vectorize_particles=True
        )

        svi = infer.SVI(model, guide, optimizer, objective)

        losses = []
        log_losses = []
        pbar = trange(num_steps)
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, n=name: gradient_norms[n].append(g.norm().item()))
        parameter_values = defaultdict(list)
        self.train()
        for idx in pbar:
            loss = svi.step(x, w)
            for name, _ in pyro.get_param_store().items():
                parameter_values[name] += [pyro.param(name).detach().cpu().numpy()]

            losses.append(loss)
            log_losses.append(np.log(losses[-1]))
            running_log_loss_mean = np.mean(log_losses[-100:])
            recent_log_loss_resid = log_losses[-100:] - running_log_loss_mean
            loss_criterion = np.max(np.abs(recent_log_loss_resid)) / running_log_loss_mean
            if idx > 100 and loss_criterion < 1e-4:
                print("Reached training convergence")
                break

            pbar.set_description(f"Loss: {losses[-1]:10.10f}")
        return losses, gradient_norms, parameter_values

def run_2d():
    K = 4
    V = 20
    W = 25
    H = 25
    R = 3
    N = 5
    NXu = 25
    eta = 0.001 / V
    constant_background = False
    permute = False

    beta = 0.01
    nplot = 1
    num_steps = 1000
    lr = 0.00005
    num_particles=15
    use_cuda = False

    topics_list = []
    words_list = []
    inferred_list = []

    for seed in range(nplot):
        pyro.get_param_store().clear()
        centers, topics, p_v_z, xs_raw, ws = generate_data_2d_circles(K=K, V=V, W=W, H=H, R=R, N=N, eta=eta, device='cuda' if use_cuda else 'cpu', permute=permute, constant_background=constant_background)
        xs = xs_raw.to(torch.get_default_dtype())
        xs[:, 0] /= W
        xs[:, 1] /= H
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        Xu = torch.rand(NXu, 2, device='cuda' if use_cuda else 'cpu')
        b = torch.tensor(beta)
        gdrf = SparseMultinomialGDRF(b=b, k=K, v=V, Xu=Xu, cuda=use_cuda)
        losses, gs, ps = gdrf.train_model(x=xs, w=ws, num_steps=num_steps, lr=lr, num_particles=num_particles)
        epochs = list(range(len(losses)))

        plt.plot(epochs, losses)
        plt.yscale('log')
        plt.title("Training loss")
        plt.show()

        gdrf_container = gdrf
        gdrf = gdrf.prior
        print(list(ps.keys()))


        mu_map = 0 #f_loc.reshape(K, W*H)
        phi_map = ps['AutoDelta.prior.phi'][-1] # beta_map.div(beta_map.sum(dim=-1, keepdim=True))
        topic_probs = gdrf.f(mu_map.detach().cpu())

        ml_topics = torch.argmax(topic_probs, dim=0)
        ml_topic_map=torch.zeros(W, H, dtype=torch.int, device='cpu')


        for idx in range(len(ml_topics)):
            x, y = xs_raw[idx, :]
            ml_topic_map[x, y] = ml_topics[idx]
        Xu_raw = 0 #posterior_Xu
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
        plot_probs(fig, ax, ps['AutoDelta.prior.phi'][0], "Initial phi")
        plt.show()
        fig, ax = plt.subplots()
        plot_probs(fig, ax, ps['AutoDelta.prior.phi'][len(ps['AutoDelta.prior.phi']) // 2], "Intermediate phi")
        plt.show()

        #
        # print(f"phi: {pyro.param('phi_q')}")
        # print(f"q: {q}")
        lengthscale = ps['prior.kernel.lengthscale']
        plt.plot(epochs, lengthscale)
        plt.title("lengthscale")
        plt.show()

run_2d()
