import torch
import torch.nn as nn
import pygmtools as pygm

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z



def log_optimal_transport(scores: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    couplings = scores

    norm = - (ms + ns).log()
    log_mu = norm.expand(m)
    log_nu = norm.expand(n)
    # log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    # log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    # print(log_mu.shape, log_nu.shape)
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SinkhornMatching(nn.Module):

    def __init__(self):
        super(SinkhornMatching, self).__init__()
        self.sinkhorn_iterations = 100
        self.eps = 1e-12
        self.bin_score = torch.tensor([0])

    def forward(self, batch_ad, batch_bd):

        # get similarity
        batch_scores = torch.einsum('bdn,bdm->bnm', batch_ad, batch_bd)
        descriptor_dim = batch_bd.shape[1]
        batch_scores = batch_scores / descriptor_dim **.5

        # Run the optimal transport.
        batch_scores = log_optimal_transport(batch_scores, iters=self.sinkhorn_iterations)

        batch_reverse_scores = torch.einsum('bdn,bdm->bnm', batch_bd, batch_ad)
        batch_reverse_scores = batch_reverse_scores / descriptor_dim **.5
        # Run the optimal transport.
        batch_reverse_scores = log_optimal_transport(batch_reverse_scores, iters=self.sinkhorn_iterations)

        return torch.exp(batch_scores), torch.exp(batch_reverse_scores)
    
if __name__ == '__main__':
    matcher = SinkhornMatching()
    batch_ad = torch.rand((1, 128, 3))
    batch_bd = torch.rand((1, 128, 3))


    batch_scores, batch_reverse_scores = matcher(batch_ad, batch_bd)
    print(batch_scores.shape, batch_reverse_scores.shape)
    print(batch_scores)