import torch


def validate_dirichlet_param(
    b: torch.Tensor, K: int, V: int, device: str = "cuda"
) -> torch.Tensor:
    assert (b <= 0).sum().detach().cpu().item() == 0, "b must be positive"
    if len(b.shape) == 0:
        assert b > 0, "b must be positive if a scalar"
        return torch.ones(K, V).to(device) * b
    elif len(b.shape) == 1:
        if b.shape[0] == K:
            return b.repeat(V, 1).T.to(device)
        elif b.shape[0] == V:
            return b.repeat(K, 1).to(device)
        else:
            raise ValueError("parameter b must have length K or V if 1D")
    elif len(b.shape) == 2:
        assert b.shape == torch.Size((K, V)), "b should be KxV if 2D"
        return b.to(device)
    else:
        raise ValueError("invalid b parameter- you passed %s", b)


def jittercholesky(Kff, N, jitter, maxjitter):
    njitter = 0
    Lff = None
    while njitter < maxjitter:
        try:
            Kff.view(-1)[:: N + 1] += jitter * (10 ** njitter)
            Lff = torch.linalg.cholesky(Kff)
            break
        except RuntimeError:
            njitter += 1
    if njitter >= maxjitter:
        raise RuntimeError("reached max jitter, covariance is unstable")
    return Lff


def generate_data_girdhar_thesis(K=5, V=100, N=8, W=150, H=32, R=8, eta=0.1, seed=777):
    K_obj = K - 1
    torch.random.manual_seed(seed)
    topics = torch.zeros((W, H), dtype=torch.int)
    centers_x = torch.randint(R, W - R, [N])
    centers_y = torch.randint(R, H - R, [N])
    centers = torch.stack([centers_x, centers_y]).T
    obj_topics = torch.randint(0, K_obj, [N])
    while len(obj_topics.unique()) < K_obj and N >= K:
        obj_topics = torch.randint(0, K_obj, [N])
    obj_topics += 1
    p_v_z = torch.tensor(
        [
            [1 + eta if V * k / K <= v < V * (k + 1) / K else eta for v in range(V)]
            for k in range(K_obj)
        ]
    )
    p_v_z /= p_v_z.sum(dim=-1, keepdim=True)  # p_v_z.shape = (K, V)]
    words = torch.zeros((W, H), dtype=torch.int)
    for w in range(W):
        for h in range(H):
            for i, center in enumerate(centers):
                diff = torch.tensor([w, h]) - center
                if (diff * diff).sum() <= R * R:
                    topics[w, h] = obj_topics[i]
            if topics[w, h] == 0:
                words[w, h] = torch.randint(0, V, [1])
            else:
                words[w, h] = torch.distributions.Categorical(
                    p_v_z[topics[w, h] - 1, :]
                ).sample()
    return centers, topics, p_v_z, words


def generate_simple_data(N=100, poisson_rate=10.0, seed=777, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    gen.manual_seed(seed)
    p = torch.rand(1, generator=gen, device=device)
    q = torch.rand(1, generator=gen, device=device)
    counts = [
        int(x)
        for x in torch.poisson(
            torch.ones(N, device=device) * poisson_rate, generator=gen
        )
    ]
    out = []
    for n in range(N):
        count = counts[n]
        probs = [q, 1 - q] if n < N * p else [1 - q, q]
        probs = torch.tensor(probs, device=device)
        cats, obs = torch.multinomial(
            probs, num_samples=count, replacement=True, generator=gen
        ).unique(return_counts=True)
        if len(cats) == 1:
            observed_cat = cats.item()
            observed_count = obs.item()
            real_obs = [0, 0]
            real_obs[observed_cat] = observed_count
            obs = torch.tensor(real_obs, dtype=torch.uint8, device=device)
        out.append(obs)
    return p, q, counts, torch.stack(out, dim=0)


def generate_data_2d_circles(
    K=5,
    V=100,
    N=8,
    W=150,
    H=32,
    R=8,
    eta=0.1,
    seed=777,
    device="cpu",
    permute=False,
    constant_background=True,
):
    """
    This function generates an artifical 2D dataset with `K` total topics and `V` total words.
    Over a background topic with a uniform distribution of words, one-topic circles are scattered.
    Each non-background topic has a word distribution a la the Girdhar thesis word distributions.
    At each pixel, a random number of observations, uniformly distributed between `V` and `10V`, is generated.
    """
    K_obj = K - 1
    gen = torch.Generator(device=device).manual_seed(seed)
    gen.manual_seed(seed)
    topics = torch.zeros((W, H), dtype=torch.int, device=device)
    centers_x = torch.randint(R, W - R, [N], device=device, generator=gen)
    centers_y = torch.randint(R, H - R, [N], device=device, generator=gen)
    centers = torch.stack([centers_x, centers_y]).T
    obj_topics = torch.randint(0, K_obj, [N], device=device, generator=gen)
    xs = torch.reshape(
        torch.stack(
            torch.meshgrid(
                torch.tensor(list(range(W)), device=device),
                torch.tensor(list(range(H)), device=device),
            )
        ),
        (2, W * H),
    ).T
    counts = torch.randint(
        low=V, high=10 * V, size=(W * H,), device=device, generator=gen
    )

    while len(obj_topics.unique()) < K_obj and N >= K:
        obj_topics = torch.randint(0, K_obj, [N], device=device, generator=gen)
    obj_topics += 1
    K_obj_probs = K_obj if constant_background else K
    p_v_z = torch.tensor(
        [
            [
                1 + eta if V * k / K_obj_probs <= v < V * (k + 1) / K_obj_probs else eta
                for v in range(V)
            ]
            for k in range(K_obj)
        ],
        device=device,
    )
    words = torch.zeros((W * H, V), dtype=torch.int, device=device)
    probs = words * 0.0
    if constant_background:
        bg_dist = torch.Tensor([1 / V for _ in range(V)]).to(probs.device)
    else:
        bg_dist = torch.Tensor(
            [1 + eta if V * (K - 1) / K <= v <= V else eta for v in range(V)]
        ).to(probs.device)
    p_v_z = torch.cat([p_v_z, bg_dist.unsqueeze(0)], dim=0)
    p_v_z /= p_v_z.sum(dim=-1, keepdim=True)  # p_v_z.shape = (K, V)]
    if permute:
        idx = torch.randperm(p_v_z.shape[1])
        p_v_z = p_v_z[:, idx].view(p_v_z.size())
    probs = probs + p_v_z[-1, :]

    for idx in range(W * H):
        w, h = xs[idx, :]
        for i in range(centers.shape[0]):
            diff = xs[idx, :] - centers[i, :]
            if (diff * diff).sum() <= R * R:
                topics[w, h] = obj_topics[i]
                probs[idx, :] = p_v_z[obj_topics[i] - 1, :]
                break
    words = torch.zeros((W * H, V), dtype=torch.int, device=device)
    for idx in range(len(counts)):
        cats, obs = torch.multinomial(
            input=probs[idx, :],
            num_samples=counts[idx],
            replacement=True,
            generator=gen,
        ).unique(return_counts=True)
        for i, cat in enumerate(cats):
            words[idx, cat] = obs[i]
    return centers, topics, p_v_z, xs, words


def generate_data_2d_uniform(V=100, W=150, H=32, seed=777, device="cpu"):
    """
    This function generates an artifical 2D dataset with `V` total words.
    Word distributions are completely uniform everywhere.
    At each pixel, a random number of observations, uniformly distributed between `V` and `10V`, is generated.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    gen.manual_seed(seed)
    xs = torch.reshape(
        torch.stack(
            torch.meshgrid(
                torch.tensor(list(range(W)), device=device),
                torch.tensor(list(range(H)), device=device),
            )
        ),
        (2, W * H),
    ).T
    counts = torch.randint(
        low=V, high=10 * V, size=(W * H,), device=device, generator=gen
    )
    words = torch.zeros((W * H, V), dtype=torch.int, device=device)
    probs = words + 1.0 / V
    for idx in range(len(counts)):
        cats, obs = torch.multinomial(
            input=probs[idx, :],
            num_samples=counts[idx],
            replacement=True,
            generator=gen,
        ).unique(return_counts=True)
        for i, cat in enumerate(cats):
            words[idx, cat] = obs[i]
    return xs, words
