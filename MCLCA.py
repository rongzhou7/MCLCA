import torch
import torch.nn as nn
import torch.nn.init as init
from math import sqrt
import math
import torch.nn.functional as F


def full_contrastive_loss(z_alpha, z_beta, tau=0.07, lambda_param=0.5):
    """
    Compute the full contrastive loss considering all negative samples explicitly,
    without normalizing by batch size.
    """
    # Normalize the embedding vectors
    z_alpha_norm = F.normalize(z_alpha, p=2, dim=1)
    z_beta_norm = F.normalize(z_beta, p=2, dim=1)

    # Calculate the cosine similarity matrix
    sim_matrix = torch.mm(z_alpha_norm, z_beta_norm.t()) / tau
    # Extract similarities of positive pairs (same index pairs)
    positive_examples = torch.diag(sim_matrix)
    # Apply exponential to the similarity matrix for negative pairs handling
    exp_sim_matrix = torch.exp(sim_matrix)
    # Create a mask to zero out positive pair contributions in negative pairs sum
    mask = torch.eye(z_alpha.size(0)).to(z_alpha.device)
    exp_sim_matrix -= mask * exp_sim_matrix
    # Sum up the exponentiated similarities for negative pairs
    negative_sum = torch.sum(exp_sim_matrix, dim=1)

    # Calculate the contrastive loss for one direction (alpha as anchor)
    L_alpha_beta = -torch.sum(torch.log(positive_examples / negative_sum))

    # Repeat the steps for the other direction (beta as anchor)
    sim_matrix_T = sim_matrix.t()
    positive_examples_T = torch.diag(sim_matrix_T)
    exp_sim_matrix_T = torch.exp(sim_matrix_T)
    exp_sim_matrix_T -= mask * exp_sim_matrix_T
    negative_sum_T = torch.sum(exp_sim_matrix_T, dim=1)
    L_beta_alpha = -torch.sum(torch.log(positive_examples_T / negative_sum_T))

    # Combine the losses from both directions, balanced by lambda
    loss = lambda_param * L_alpha_beta + (1 - lambda_param) * L_beta_alpha
    return loss  # Return the unnormalized total loss


def contrastive_loss(z_alpha, z_beta, lambda_param, tau=0.07):
    """
    Compute the contrastive loss L_cont(α, β) for two sets of embeddings.

    Parameters:
    - z_alpha: Embeddings from modality α, tensor of shape (batch_size, embedding_size)
    - z_beta: Embeddings from modality β, tensor of shape (batch_size, embedding_size)
    - tau: Temperature parameter for scaling the cosine similarity
    - lambda_param: Weighting parameter to balance the loss terms

    Returns:
    - loss: The computed contrastive loss
    """

    # Compute the cosine similarity matrix
    sim_matrix = torch.mm(F.normalize(z_alpha, p=2, dim=1), F.normalize(z_beta, p=2, dim=1).t()) / tau
    # Diagonal elements are positive examples
    positive_examples = torch.diag(sim_matrix)
    # Compute the log-sum-exp for the denominator
    sum_exp = torch.logsumexp(sim_matrix, dim=1)

    # Loss for one direction (α anchoring and contrasting β)
    L_alpha_beta = -torch.mean(positive_examples - sum_exp)

    # Loss for the other direction (β anchoring and contrasting α)
    L_beta_alpha = -torch.mean(torch.diag(
        torch.mm(F.normalize(z_beta, p=2, dim=1), F.normalize(z_alpha, p=2, dim=1).t()) / tau) - torch.logsumexp(
        sim_matrix.t(), dim=1))

    # Combined loss, balanced by lambda
    loss = lambda_param * L_alpha_beta + (1 - lambda_param) * L_beta_alpha

    return loss

def cross_modal_attention(q, k, v, dk):
    """
    Calculate cross-modal attention between two modalities.

    Parameters:
    - q: Query matrix from modality α, shape (batch_size, embedding_dim)
    - k: Key matrix from modality β, shape (batch_size, embedding_dim)
    - v: Value matrix from modality β, shape (batch_size, embedding_dim)
    - dk: Dimension of the key space (typically embedding_dim), used to scale the dot products

    Returns:
    - attention_output: Updated representation for modality α based on information from modality β
    """
    # Compute the dot products between query and key matrices
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Multiply the attention weights by the value matrix to get the final attention output
    attention_output = torch.matmul(attention_weights, v)

    return attention_output

def self_attention(query, key, value, dropout=None, mask=None):
    """
    Compute self-attention scores and apply it to the value.
    Args:
        query (Tensor): Query tensor 'Q'.
        key (Tensor): Key tensor 'K'.
        value (Tensor): Value tensor 'V'.
        dropout (function, optional): Dropout function to be applied to attention scores.
        mask (Tensor, optional): Mask tensor to mask certain positions before softmax.
    Returns:
        Tuple[Tensor, Tensor]: Output after applying self-attention and the attention scores.
    """
    # Dimension of key to scale down dot product values
    d_k = query.size(-1)
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    # 判断是否要mask，注：mask的操作在QK之后，softmax之前
    # if mask is not None:
    #     """
    #     scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
    #     在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
    #     """
    #     # mask.cuda()
    #     # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引
    #
    #   scores = scores.masked_fill(mask == 0, -1e9)
    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
    # # Apply dropout if provided
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)

    return torch.matmul(self_attn_softmax, value), self_attn_softmax

# Weight & Bias Initialization
def initialization(net):
    """
        Initialize weights and biases of a neural network.
        Args:
            net (nn.Module): Neural network module.
    """
    if isinstance(net, nn.Linear):
        init.xavier_uniform(net.weight)
        init.zeros_(net.bias)


import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(CrossModalAttention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

    def forward(self, h1, h2, h3, h4):
        # Compute queries, keys, values for all modalities
        Q1, K1, V1 = self.query(h1), self.key(h1), self.value(h1)
        Q2, K2, V2 = self.query(h2), self.key(h2), self.value(h2)
        Q3, K3, V3 = self.query(h3), self.key(h3), self.value(h3)
        Q4, K4, V4 = self.query(h4), self.key(h4), self.value(h4)

        # Calculate attention scores between different modalities
        attention_scores_12 = torch.matmul(Q1, K2.t())
        attention_scores_13 = torch.matmul(Q1, K3.t())
        attention_scores_14 = torch.matmul(Q1, K4.t())
        attention_12 = torch.matmul(F.softmax(attention_scores_12, dim=-1), V2)
        attention_13 = torch.matmul(F.softmax(attention_scores_13, dim=-1), V3)
        attention_14 = torch.matmul(F.softmax(attention_scores_14, dim=-1), V4)

        attention_scores_21 = torch.matmul(Q2, K1.t())
        attention_scores_23 = torch.matmul(Q2, K3.t())
        attention_scores_24 = torch.matmul(Q2, K4.t())
        attention_21 = torch.matmul(F.softmax(attention_scores_21, dim=-1), V1)
        attention_23 = torch.matmul(F.softmax(attention_scores_23, dim=-1), V3)
        attention_24 = torch.matmul(F.softmax(attention_scores_24, dim=-1), V4)

        attention_scores_31 = torch.matmul(Q3, K1.t())
        attention_scores_32 = torch.matmul(Q3, K2.t())
        attention_scores_34 = torch.matmul(Q3, K4.t())
        attention_31 = torch.matmul(F.softmax(attention_scores_31, dim=-1), V1)
        attention_32 = torch.matmul(F.softmax(attention_scores_32, dim=-1), V2)
        attention_34 = torch.matmul(F.softmax(attention_scores_34, dim=-1), V4)

        attention_scores_41 = torch.matmul(Q4, K1.t())
        attention_scores_42 = torch.matmul(Q4, K2.t())
        attention_scores_43 = torch.matmul(Q4, K3.t())
        attention_41 = torch.matmul(F.softmax(attention_scores_41, dim=-1), V1)
        attention_42 = torch.matmul(F.softmax(attention_scores_42, dim=-1), V2)
        attention_43 = torch.matmul(F.softmax(attention_scores_43, dim=-1), V3)

        # Sum of attention weighted values for each modality
        r1= attention_12 + attention_13 + attention_14
        r2= attention_21 + attention_23 + attention_24
        r3= attention_31 + attention_32 + attention_34
        r4= attention_41 + attention_42 + attention_43

        return r1, r2, r3, r4  # Return updated representations

# modality3 is SNP
class MCLCA(nn.Module):
    """
       ADCCA model for four different data modalities.
       Each modality has its own neural network, and the model applies self-attention
       to each modality's output.
    """
    def __init__(self, m1_embedding_list, m2_embedding_list, m3_embedding_list, m4_embedding_list, lambda_param):
        super(MCLCA, self).__init__()
        # Embedding List of each modality
        m1_du0, m1_du1, m1_du2, m1_du3 = m1_embedding_list
        m2_du0, m2_du1, m2_du2, m2_du3 = m2_embedding_list
        m3_du0, m3_du1, m3_du2 = m3_embedding_list
        m4_du0, m4_du1, m4_du2, m4_du3 = m4_embedding_list

        # Initialize neural networks for each modality
        self.f1 = nn.Sequential(
            nn.Linear(m1_du0, m1_du1), nn.Tanh(),
            nn.Linear(m1_du1, m1_du2), nn.Tanh(),
            nn.Linear(m1_du2, m1_du3), nn.Tanh())

        self.f2 = nn.Sequential(
            nn.Linear(m2_du0, m2_du1), nn.Tanh(),
            nn.Linear(m2_du1, m2_du2), nn.Tanh(),
            nn.Linear(m2_du2, m2_du3), nn.Tanh())

        self.f3 = nn.Sequential(
            nn.Linear(m3_du0, m3_du1), nn.Tanh(),
            nn.Linear(m3_du1, m3_du2), nn.Tanh())

        self.f4 = nn.Sequential(
            nn.Linear(m4_du0, m4_du1), nn.Tanh(),
            nn.Linear(m4_du1, m4_du2), nn.Tanh(),
            nn.Linear(m4_du2, m4_du3), nn.Tanh())


        # Weight & Bias Initialization
        self.f1.apply(initialization)
        self.f2.apply(initialization)
        self.f3.apply(initialization)
        self.f4.apply(initialization)

        self.g1 = nn.Sequential(
            nn.Linear(m1_du3, 64), nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.g2 = nn.Sequential(
            nn.Linear(m2_du3, 64), nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.g3 = nn.Sequential(
            nn.Linear(m3_du2, 64), nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.g4 = nn.Sequential(
            nn.Linear(m4_du3, 64), nn.ReLU(),
            nn.Linear(64, 128)
        )

        self.g1.apply(initialization)
        self.g2.apply(initialization)
        self.g3.apply(initialization)
        self.g4.apply(initialization)

        self.lambda_param = lambda_param

        self.cross_modal_attention = CrossModalAttention(16, 16)

        self.mlp = nn.Sequential(
            nn.Linear(16 * 4, 256),  # Assuming each r dimension is 16*4
            nn.ReLU(),
            nn.Linear(256, 2)  # Only one output neuron
        )

        self.sigmoid = nn.Sigmoid()


        # self.softmax = nn.Softmax(dim=1)

    # Compute outputs for each modality with self-attention
    def forward(self, x1, x2, x3, x4):
        h1 = self.f1(x1)
        h2 = self.f2(x2)
        h3 = self.f3(x3)
        h4 = self.f4(x4)

        z1 = self.g1(h1)
        z2 = self.g2(h2)
        z3 = self.g3(h3)
        z4 = self.g4(h4)

        # Compute the contrastive loss between each pair of projections
        loss12 = full_contrastive_loss(z1, z2)
        loss13 = full_contrastive_loss(z1, z3)
        loss14 = full_contrastive_loss(z1, z4)
        loss23 = full_contrastive_loss(z2, z3)
        loss24 = full_contrastive_loss(z2, z4)
        loss34 = full_contrastive_loss(z3, z4)

        # Combine losses, could average or sum depending on your preference
        total_contrastive_loss = (loss12 + loss13 + loss14 + loss23 + loss24 + loss34)

        r1, r2, r3, r4 = self.cross_modal_attention(h1, h2, h3, h4)

        r_concat = torch.cat([r1, r2, r3, r4], dim=1)

        y_hat = self.sigmoid(self.mlp(r_concat))

        return total_contrastive_loss

    def predict(self, x1, x2, x3, x4):
        h1 = self.f1(x1)
        h2 = self.f2(x2)
        h3 = self.f3(x3)
        h4 = self.f4(x4)

        r1, r2, r3, r4 = self.cross_modal_attention(h1, h2, h3, h4)

        r_concat = torch.cat([r1, r2, r3, r4], dim=1)

        y_hat = self.sigmoid(self.mlp(r_concat))

        return y_hat
