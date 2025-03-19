import torch
import torch.nn as nn
import torch.nn.functional as F

def energy_loss(energy_net, batch, alpha=0.1, margin=0.5):
    print('insider energy loss')
    """Compute energy loss with margin ranking."""
    trajectories = batch['trajectory']
    features = batch['features']
    true_energies = batch['energy']
    
    pred_energies = energy_net(trajectories, features).squeeze()
    mse_loss = F.mse_loss(pred_energies, true_energies)
    
    ranking_loss = 0
    for i in range(len(pred_energies)):
        print("inside loop of energy loss")
        for j in range(i + 1, len(pred_energies)):
            if true_energies[i] < true_energies[j]:
                ranking_loss += F.relu(
                    -(pred_energies[j] - pred_energies[i]) + margin
                )
    
    ranking_loss = ranking_loss / (len(pred_energies) * (len(pred_energies) - 1) / 2)
    return mse_loss + alpha * ranking_loss



def flow_matching_loss(model, x0, x1, energy_net=None):
    """Compute flow matching loss with optional energy regularization."""
    batch_size = x0.shape[0]
    t = torch.rand(batch_size, device=x0.device)
    
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0 #this is the true velocity 
    
    preds, mode_logits = model(xt, t) #this is the predicted velocity
    mode_losses = []
    
    for pred in preds:
        mode_losses.append(((target - pred)**2).mean(dim=(1, 2))) #mse of true and predicted velocity
    
    mode_losses = torch.stack(mode_losses, dim=1)
    min_losses = torch.min(mode_losses, dim=1)[0]
    
    if energy_net is not None:
        energy_loss = energy_net(xt).mean()
        return min_losses.mean() + 0.1 * energy_loss
    
    return min_losses.mean()

