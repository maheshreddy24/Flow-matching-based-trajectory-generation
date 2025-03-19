import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class TimeEmbedding(nn.Module):
    """Time embedding layer for the flow matching model."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
    def forward(self, t, max_positions=10000):
        half_dim = self.channels // 2
        emb = torch.log(torch.tensor(max_positions)) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.channels % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb

class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Linear(channels, channels)
        self.conv2 = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + identity

class EnhancedEnergyNetwork(nn.Module):
    """Neural network for learning trajectory energies."""
    def __init__(self, input_dim=3, hidden_dim=256, feature_dim=4):
        super().__init__()
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim)
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Ensure non-negative energy
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
    
    def forward(self, x, features=None):
        print("inside energy network")
        batch_size = x.shape[0]
        
        # Encode trajectory
        traj_features = self.trajectory_encoder(x)
        
        # Apply self-attention
        traj_features = traj_features.transpose(0, 1)
        attended_features, _ = self.attention(traj_features, traj_features, traj_features)
        attended_features = attended_features.transpose(0, 1)
        
        # Global pooling
        traj_features = torch.mean(attended_features, dim=1)
        
        if features is not None:
            feat_embedding = self.feature_encoder(features)
            combined_features = torch.cat([traj_features, feat_embedding], dim=1)
            features = self.fusion_layer(combined_features)
        else:
            features = traj_features
        
        energy = self.energy_head(features)
        return energy

class MultiModalFlowMatching(nn.Module):
    """Multi-modal flow matching model for trajectory generation."""
    def __init__(self, input_dim=3, hidden_dim=512, num_layers=8):
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modes = 3
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        self.mode_proj = nn.Sequential(
            nn.Linear(self.num_modes, hidden_dim),
            nn.GELU()
        )
        
        self.mode_selection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_modes)
        )
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t, mode_indices=None):
        batch_size = x.shape[0]
        
        h = self.input_proj(x)
        t_emb = self.time_embed(t)
        t_emb = self.time_proj(t_emb)
        h = h + t_emb.unsqueeze(1)
        
        for block in self.blocks:
            h = block(h)
        
        mode_logits = self.mode_selection(torch.mean(h, dim=1))
        
        if mode_indices is None:
            outputs = []
            for i in range(self.num_modes):
                mode_embedding = F.one_hot(
                    torch.full((batch_size,), i, device=h.device), 
                    num_classes=self.num_modes
                ).float()
                mode_h = h + self.mode_proj(mode_embedding).unsqueeze(1)
                outputs.append(self.output_proj(mode_h))
            return outputs, mode_logits
        else:
            mode_embedding = F.one_hot(mode_indices, num_classes=self.num_modes).float()
            mode_h = h + self.mode_proj(mode_embedding).unsqueeze(1)
            return self.output_proj(mode_h)


def visualize_energy_predictions(energy_net, datasets, save_path):
    """Generate visualization of energy predictions."""
    energy_net.eval()
    predictions = {}
    
    with torch.no_grad():
        for name, trajectories in datasets.items():
            energies = []
            for traj in trajectories:
                features = traj.unsqueeze(0)
                energy = energy_net(features).item()
                energies.append(energy)
            predictions[name] = energies
    
    plt.figure(figsize=(10, 6))
    for name, energies in predictions.items():
        plt.hist(energies, label=name, alpha=0.6, bins=30)
    
    plt.xlabel('Predicted Energy')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Energies')
    plt.legend()
    plt.savefig(save_path)
    plt.close()




def train_energy_network(energy_net, train_loader, val_loader, num_epochs, learning_rate, device, save_dir):
    """Train the energy network."""
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    print("inside train_energy network")
    for epoch in range(num_epochs):
        energy_net.train()
        train_losses = []
        
        for batch in train_loader:
            # Assuming batch contains trajectory, features, and energy in that order
            trajectory, features, energy = batch
            batch_dict = {
                'trajectory': trajectory.to(device),
                'features': features.to(device),
                'energy': energy.to(device)
            }
            
            optimizer.zero_grad()
            loss = energy_loss(energy_net, batch_dict)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        energy_net.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                trajectory, features, energy = batch
                batch_dict = {
                    'trajectory': trajectory.to(device),
                    'features': features.to(device),
                    'energy': energy.to(device)
                }
                loss = energy_loss(energy_net, batch_dict)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(energy_net.state_dict(), save_dir / 'best_energy_net.pt')


from trajectory import generate_datasets
from trajectory import EnhancedTrajectoryGenerator, EnergyDatasetGenerator, EnhancedEnergyDataset


def main():
    """Main training script."""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    flow_model = MultiModalFlowMatching(input_dim=3).to(device)
    energy_net = EnhancedEnergyNetwork(input_dim=3).to(device)
    
    # Generate initial datasets for flow model training
    print("Generating initial datasets for flow training...")
    datasets = generate_datasets(num_trajectories=10000, cosine_threshold=0.7)
    
    # Convert pandas DataFrames to torch tensors for flow model training
    def convert_to_tensor(trajectory_df):
        return torch.tensor(trajectory_df[['x', 'y', 'z']].values, dtype=torch.float32)
    
    flow_train_data = [convert_to_tensor(traj) for traj in datasets['good_trajectories']]
    
    # Create flow model training loader
    flow_dataset = torch.utils.data.TensorDataset(
        torch.stack(flow_train_data)
    )
    
    flow_train_loader = DataLoader(
        flow_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path('experiments') / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train flow model first
    print("\nTraining flow model...")
    flow_optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-4)
    
    for epoch in range(50):  # Train for 50 epochs
        flow_model.train()
        epoch_losses = []
        
        for batch in flow_train_loader:
            trajectories = batch[0].to(device)
            x0 = torch.randn_like(trajectories)  # Initial noise
            x1 = trajectories  # Target trajectories
            
            flow_optimizer.zero_grad()
            from losses import flow_matching_loss
            loss = flow_matching_loss(flow_model, x0, x1) #this is the optimsation function, which takes the random noise and final distrinution
            loss.backward()
            flow_optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        print(f"Flow Model - Epoch {epoch+1}/50, Loss: {avg_loss:.4f}")
    
    # Save trained flow model
    torch.save(flow_model.state_dict(), save_dir / 'trained_flow_model.pt')
    
    # Now use trained flow model to generate trajectories for energy network
    print("\nGenerating datasets for energy network training...")
    generator = EnhancedTrajectoryGenerator()
    dataset_gen = EnergyDatasetGenerator(flow_model, generator)
    
    # Generate trajectories using trained flow model
    good_trajectories = dataset_gen.generate_good_trajectories(1000, device)
    print(f"Generated {len(good_trajectories)} good trajectories")
    
    perturbed_trajectories = dataset_gen.generate_perturbed_trajectories(good_trajectories)
    print(f"Generated {len(perturbed_trajectories)} perturbed trajectories")
    
    random_trajectories = dataset_gen.generate_random_trajectories(1000)
    print(f"Generated {len(random_trajectories)} random trajectories")
    
    # Create energy network dataset
    dataset = EnhancedEnergyDataset(good_trajectories, perturbed_trajectories, random_trajectories)
    
    # Split dataset for energy network training
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Save configuration
    config = {
        'device': str(device),
        'train_size': train_size,
        'val_size': val_size,
        'good_trajectories': len(good_trajectories),
        'perturbed_trajectories': len(perturbed_trajectories),
        'random_trajectories': len(random_trajectories),
        'timestamp': timestamp
    }
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train energy network
    print("\nTraining energy network...")
    train_energy_network(
        energy_net=energy_net,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=1e-4,
        device=device,
        save_dir=save_dir
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_energy_predictions(
        energy_net=energy_net,
        datasets={
            'Good': good_trajectories,
            'Perturbed': perturbed_trajectories,
            'Random': random_trajectories
        },
        save_path=save_dir / 'energy_distributions.png'
    )
    
    # Save final models
    torch.save({
        'energy_net_state_dict': energy_net.state_dict(),
        'flow_model_state_dict': flow_model.state_dict(),
        'config': config
    }, save_dir / 'final_models.pt')
    
    # Plot example trajectories
    print("\nPlotting example trajectories...")
    energy_net.eval()
    with torch.no_grad():
        fig = plt.figure(figsize=(15, 5))
        categories = ['Good', 'Perturbed', 'Random']
        trajectories = [good_trajectories[0], perturbed_trajectories[0], random_trajectories[0]]
        
        for i, (category, traj) in enumerate(zip(categories, trajectories)):
            features = dataset._compute_trajectory_features(traj.unsqueeze(0))
            energy = energy_net(traj.unsqueeze(0).to(device), features.to(device)).item()
            
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            traj = traj.cpu().numpy()
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
            ax.set_title(f'{category}\nEnergy: {energy:.2f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'example_trajectories.png')
        plt.close()
    
    print("\nDone!")

if __name__ == "__main__":
    main()