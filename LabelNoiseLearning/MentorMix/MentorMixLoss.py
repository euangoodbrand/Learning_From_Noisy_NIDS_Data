import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as cat
import torch.distributions.dirichlet as diri

def MentorMixLoss(args, MentorNet, StudentNet, x_i, y_i, v_true, loss_p_prev, loss_p_second_prev, epoch):
    XLoss = torch.nn.CrossEntropyLoss(reduction='none')
    bsz = x_i.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_i, y_i, v_true = x_i.to(device), y_i.to(device), v_true.to(device)

    with torch.no_grad():
        outputs_i = StudentNet(x_i)
        loss = F.cross_entropy(outputs_i, y_i, reduction='none')
        sorted_losses = torch.sort(loss).values
        loss_p = args.ema * loss_p_prev + (1 - args.ema) * sorted_losses[int(bsz * args.gamma_p)]
        loss_diff = loss - loss_p
        v = MentorNet(v_true, args.n_epoch, epoch, loss, loss_diff)

        if epoch < int(args.n_epoch * 0.2):
            v = torch.bernoulli(torch.ones_like(loss_diff) / 2).to(device)

    P_v = cat.Categorical(F.softmax(v, dim=0))
    indices_j = P_v.sample((bsz,))

    x_j = x_i[indices_j]
    y_j = y_i[indices_j]

    Beta = diri.Dirichlet(torch.tensor([args.alpha] * 2).to(device))
    lambdas = Beta.sample([bsz]).to(device)
    lambdas_max = lambdas.max(dim=1)[0]
    lambdas = v * lambdas_max + (1 - v) * (1 - lambdas_max)

    lambdas_expanded = lambdas.view(bsz, 1).expand_as(x_i)
    x_tilde = x_i * lambdas_expanded + x_j * (1 - lambdas_expanded)
    outputs_tilde = StudentNet(x_tilde)

    print("x_i shape:", x_i.shape)
    print("y_i shape:", y_i.shape)
    print("outputs_i shape:", outputs_i.shape)
    print("Before mixup - x_i shape:", x_i.shape)
    print("Before mixup - x_j shape:", x_j.shape)
    print("Lambdas shape:", lambdas.shape)
    print("After mixup - x_tilde shape:", x_tilde.shape)
    print("After mixup - outputs_tilde shape:", outputs_tilde.shape)

    # Ensure the outputs are in the correct shape
    if outputs_tilde.dim() > 2:
        outputs_tilde = outputs_tilde.view(bsz, -1)

    print("outputs_tilde shape after view:", outputs_tilde.shape)

    mixed_loss_i = XLoss(outputs_tilde, y_i)
    mixed_loss_j = XLoss(outputs_tilde, y_j)
    final_loss = lambdas * mixed_loss_i + (1 - lambdas) * mixed_loss_j

    with torch.no_grad():
        sorted_final_loss = torch.sort(final_loss).values
        loss_p_second = args.ema * loss_p_second_prev + (1 - args.ema) * sorted_final_loss[int(bsz * args.gamma_p)]
        loss_diff = final_loss - loss_p_second
        v_mix = MentorNet(v_true, args.n_epoch, epoch, final_loss, loss_diff)

        if epoch < int(args.n_epoch * 0.2):
            v_mix = torch.bernoulli(torch.ones_like(loss_diff) / 2).to(device)

    final_loss = final_loss * v_mix

    return final_loss.mean(), loss_p, loss_p_second, v