import torch
import torch.nn as nn
import torch.nn.functional as F


class MCL(nn.Module):
    def __init__(self, input_dim, hidden_dim, projection_dim, num_modalities):
        super(MCL, self).__init__()
        self.num_modalities = num_modalities
        self.f_m = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_modalities)
        ])

        self.g_m = nn.ModuleList([

            nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU()
        ) for _ in range(num_modalities)])

        self.tau = 0.05
        self.lambda_param = 0.5

    def forward(self, x):
        h_m = [self.f_m[i](x[i]) for i in range(self.num_modalities)]
        z_m = [self.g_m[i](h_m[i]) for i in range(self.num_modalities)]

        loss_mcl = 0
        for alpha in range(self.num_modalities - 1):
            for beta in range(alpha + 1, self.num_modalities):
                loss_mcl += self.contrastive_loss(z_m[alpha], z_m[beta])

        return h_m, loss_mcl

    def contrastive_loss(self, z_alpha, z_beta):
        N = z_alpha.size(0)
        cos_sim = torch.nn.functional.cosine_similarity(z_alpha.unsqueeze(1), z_beta.unsqueeze(0), dim=2)
        loss_alpha_beta = -torch.log(torch.exp(cos_sim / self.tau).diag() / (
                    torch.exp(cos_sim / self.tau).sum(dim=1) - torch.exp(cos_sim.diag() / self.tau)))
        loss_beta_alpha = -torch.log(torch.exp(cos_sim / self.tau).diag() / (
                    torch.exp(cos_sim / self.tau).sum(dim=0) - torch.exp(cos_sim.diag() / self.tau)))
        loss_cont = self.lambda_param * loss_alpha_beta.mean() + (1 - self.lambda_param) * loss_beta_alpha.mean()
        return loss_cont


class CAD(nn.Module):
    def __init__(self, hidden_dim, dk, dv, output_dim, num_modalities):
        super(CAD, self).__init__()
        self.num_modalities = num_modalities
        self.W_q = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, dk)) for _ in range(num_modalities)])
        self.W_k = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, dk)) for _ in range(num_modalities)])
        self.W_v = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, dv)) for _ in range(num_modalities)])

        self.fc = nn.Linear(num_modalities * dv, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h_m):
        Q = [torch.matmul(h_m[i], self.W_q[i]) for i in range(self.num_modalities)]
        K = [torch.matmul(h_m[i], self.W_k[i]) for i in range(self.num_modalities)]
        V = [torch.matmul(h_m[i], self.W_v[i]) for i in range(self.num_modalities)]

        r_concat = []
        for alpha in range(self.num_modalities):
            r_alpha = sum(
                [self.cross_modal_attention(Q[alpha], K[beta], V[alpha]) for beta in range(self.num_modalities) if
                 beta != alpha])
            r_concat.append(r_alpha)

        r_concat = torch.cat(r_concat, dim=1)
        y_hat = self.fc(r_concat)
        y_hat = self.softmax(y_hat)

        return y_hat

    def cross_modal_attention(self, Q_alpha, K_beta, V_alpha):
        score = torch.matmul(Q_alpha, K_beta.transpose(1, 2)) / torch.sqrt(K_beta.size(2))
        score = torch.nn.functional.softmax(score, dim=2)
        h_prime_alpha = torch.matmul(score, V_alpha)
        return h_prime_alpha


class MultimodalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, projection_dim, dk, dv, output_dim, num_modalities):
        super(MultimodalNetwork, self).__init__()
        self.mcl_module = MCL(input_dim, hidden_dim, projection_dim, num_modalities)
        self.cad_module = CAD(hidden_dim, dk, dv, output_dim, num_modalities)
        self.gamma = 1.0

    def forward(self, x, y, mask):
        h_m, loss_mcl = self.mcl_module(x)
        y_hat = self.cad_module(h_m)

        loss_cls = -torch.sum(mask * y * torch.log(y_hat))
        loss = loss_mcl + self.gamma * loss_cls

        return y_hat, loss

# Instantiate and use the model
# model = MultimodalNetwork(input_dim, hidden_dim, projection_dim, dk, dv, output_dim, num_modalities)
# y_hat, loss = model(x, y, mask)
