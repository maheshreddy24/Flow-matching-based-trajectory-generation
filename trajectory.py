import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
from scipy.spatial.distance import cosine



#this is a simple class with all types of trajectory generatio methods
class EnhancedTrajectoryGenerator:
    """A class to generate various types of trajectories in 3D space."""
    
    def __init__(
        self,
        time_duration: float = 1.0,
        num_points: int = 50,
        workspace_bounds: dict = {
            'x': (-1.0, 1.0),
            'y': (-1.0, 1.0),
            'z': (0.0, 2.0)
        }
    ):
        """
        Initialize the trajectory generator.
        
        Args:
            time_duration: Duration of the trajectory in seconds
            num_points: Number of points in the trajectory
            workspace_bounds: Dictionary defining the workspace boundaries
        """
        self.time_duration = time_duration
        self.num_points = num_points
        self.workspace_bounds = workspace_bounds
        self.t = np.linspace(0, time_duration, num_points)
    
    def _generate_parabolic(self, start_height, end_height, distance):
        """Generate a parabolic trajectory."""
        x = distance * (self.t / self.time_duration)
        y = np.zeros_like(self.t)
        height_diff = end_height - start_height
        z = (
            start_height + 
            height_diff * (self.t / self.time_duration) +
            4 * height_diff * (self.t / self.time_duration) * (1 - self.t / self.time_duration)
        )
        return x, y, z
    
    def _generate_circular_arc(self, radius, start_angle, end_angle, height):
        """Generate a circular arc trajectory."""
        angles = np.linspace(start_angle, end_angle, self.num_points)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = height * np.ones_like(x)
        return x, y, z
    
    def _generate_spiral(self, radius, height_range, revolutions):
        """Generate a spiral trajectory."""
        angles = np.linspace(0, 2*np.pi*revolutions, self.num_points)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.linspace(height_range[0], height_range[1], self.num_points)
        return x, y, z
    
    def _generate_s_curve(self, amplitude, frequency, height):
        """Generate an S-curve trajectory."""
        x = self.t * (self.workspace_bounds['x'][1] - self.workspace_bounds['x'][0])
        y = amplitude * np.sin(frequency * self.t)
        z = height * np.ones_like(x)
        return x, y, z
    
    def _generate_zigzag(self, num_zigs, amplitude, height):
        """Generate a zigzag trajectory."""
        x = self.t * (self.workspace_bounds['x'][1] - self.workspace_bounds['x'][0])
        y = amplitude * np.abs((self.t * num_zigs * 2) % 2 - 1)
        z = height * np.ones_like(x)
        return x, y, z
    
    def _generate_diagonal(self, start_point, end_point):
        """Generate a diagonal straight line trajectory."""
        x = np.linspace(start_point[0], end_point[0], self.num_points)
        y = np.linspace(start_point[1], end_point[1], self.num_points)
        z = np.linspace(start_point[2], end_point[2], self.num_points)
        return x, y, z
    
    def generate_trajectory(self, traj_type: str, params: dict, noise_std: float = 0.02) -> pd.DataFrame:
        """
        Generate a single trajectory of specified type.
        
        Args:
            traj_type: Type of trajectory to generate
            params: Parameters for the trajectory generation
            noise_std: Standard deviation of noise to add
            
        Returns:
            DataFrame containing the trajectory points
        """
        trajectory_generators = {
            "parabolic": lambda: self._generate_parabolic(
                params.get('start_height', 0.1),
                params.get('end_height', 1.5),
                params.get('distance', 0.6)
            ),
            "circular": lambda: self._generate_circular_arc(
                params.get('radius', 0.5),
                params.get('start_angle', 0),
                params.get('end_angle', np.pi),
                params.get('height', 1.0)
            ),
            "spiral": lambda: self._generate_spiral(
                params.get('radius', 0.3),
                params.get('height_range', (0.1, 1.5)),
                params.get('revolutions', 2)
            ),
            "s_curve": lambda: self._generate_s_curve(
                params.get('amplitude', 0.3),
                params.get('frequency', 3),
                params.get('height', 1.0)
            ),
            "zigzag": lambda: self._generate_zigzag(
                params.get('num_zigs', 4),
                params.get('amplitude', 0.3),
                params.get('height', 1.0)
            ),
            "diagonal": lambda: self._generate_diagonal(
                params.get('start_point', [0, 0, 0.1]),
                params.get('end_point', [1, 1, 1.5])
            )
        }
        
        if traj_type not in trajectory_generators:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
            
        x, y, z = trajectory_generators[traj_type]()
            
        # Add noise if specified
        if noise_std > 0:
            x += np.random.normal(0, noise_std, size=len(x))
            y += np.random.normal(0, noise_std, size=len(y))
            z += np.random.normal(0, noise_std, size=len(z))
            
        return pd.DataFrame({'t': self.t, 'x': x, 'y': y, 'z': z})


def generate_datasets(num_trajectories: int = 10000, cosine_threshold: float = 0.7):
    """
    Generate training datasets for both flow matching and energy networks.
    
    Args:
        num_trajectories: Number of trajectories to generate
        cosine_threshold: Threshold for filtering similar trajectories
        
    Returns:
        Dictionary containing good, perturbed, and random trajectories
    """
    generator = EnhancedTrajectoryGenerator() #this is an instance of class Trjaecoty generator
    
    # List of trajectory types and their parameter ranges
    traj_configs = [
        ("parabolic", {
            'start_height': (0.3, 0.3),
            'end_height': (1.2, 1.5),
            'distance': (0.4, 0.8)
        }),
        ("circular", {
            'radius': (0.2, 0.7),
            'start_angle': (0, np.pi/4),
            'end_angle': (3*np.pi/4, np.pi),
            'height': (0.5, 1.5)
        }),
        ("spiral", {
            'radius': (0.2, 0.4),
            'height_range': [(0.1, 0.3), (1.2, 1.5)],
            'revolutions': (1.5, 2.5)
        }),
        ("s_curve", {
            'amplitude': (0.1, 0.4),
            'frequency': (2, 4),
            'height': (0.8, 1.2)
        }),
        ("zigzag", {
            'num_zigs': (3, 5),
            'amplitude': (0.2, 0.4),
            'height': (0.8, 1.2)
        }),
        ("diagonal", {
            'start_point': [(0, 0, 0.1), (0.2, 0.2, 0.3)],
            'end_point': [(0.8, 0.8, 1.2), (1.0, 1.0, 1.5)]
        })
    ]
    
    good_trajectories = []
    for _ in range(num_trajectories):
        traj_type, param_ranges = random.choice(traj_configs)
        params = {
            k: random.uniform(*v) if isinstance(v, tuple) else random.choice(v)
            for k, v in param_ranges.items()
        }
        traj = generator.generate_trajectory(traj_type, params, noise_std=0.02)
        good_trajectories.append(traj)
    
    # Filter similar trajectories
    filtered_trajectories = []
    for i, traj1 in enumerate(good_trajectories):
        is_unique = True
        traj1_vec = traj1[['x', 'y', 'z']].values.flatten()
        
        for j in range(i):
            traj2 = good_trajectories[j]
            traj2_vec = traj2[['x', 'y', 'z']].values.flatten()
            similarity = 1 - cosine(traj1_vec, traj2_vec)
            
            if similarity > cosine_threshold:
                is_unique = False
                break
                
        if is_unique:
            filtered_trajectories.append(traj1)
    
    # Generate variations
    perturbed_trajectories = []
    random_trajectories = []
    
    for traj in filtered_trajectories:
        # Add perturbations
        perturbed = traj.copy()
        noise = np.random.normal(0, 0.1, size=(len(traj), 3))
        perturbed[['x', 'y', 'z']] += noise
        perturbed_trajectories.append(perturbed)
        
        # Generate random trajectory
        random_traj = pd.DataFrame({
            't': generator.t,
            'x': np.random.normal(0, 0.5, size=generator.num_points),
            'y': np.random.normal(0, 0.5, size=generator.num_points),
            'z': np.random.normal(0.5, 0.5, size=generator.num_points)
        })
        random_trajectories.append(random_traj)
    
    return {
        'good_trajectories': filtered_trajectories,
        'perturbed_trajectories': perturbed_trajectories,
        'random_trajectories': random_trajectories
    } ##this returns  a dict of 3 kinds of trajectories filtered, perturbed, random


class EnergyDatasetGenerator:
    """Generator for energy-based trajectory datasets."""
    
    def __init__(self, flow_model, trajectory_generator):
        """
        Initialize the energy dataset generator.
        
        Args:
            flow_model: Trained flow model for generating trajectories
            trajectory_generator: Instance of EnhancedTrajectoryGenerator
        """
        self.flow_model = flow_model
        self.generator = trajectory_generator
        
    def generate_good_trajectories(self, num_samples, device='cpu'):
        """Generate good trajectories using trained flow model."""
        trajectories = []
        self.flow_model.eval()
        with torch.no_grad():
            for _ in range(num_samples):
                x0 = torch.randn(1, 50, 3).to(device)
                t = torch.ones(1).to(device)
                for mode in range(self.flow_model.num_modes):
                    traj = self.flow_model(x0, t, mode_indices=torch.tensor([mode]).to(device))
                    trajectories.append(traj.cpu().squeeze(0))
        
        return torch.stack(trajectories)
    
    def generate_perturbed_trajectories(self, good_trajectories, perturbation_types=['noise', 'time_shift', 'spatial_shift']):
        """Generate perturbed versions of good trajectories."""
        perturbed = []
        for traj in good_trajectories:
            for _ in range(2):
                perturb_type = np.random.choice(perturbation_types)
                if perturb_type == 'noise':
                    noise = torch.randn_like(traj) * 0.1
                    perturbed.append(traj + noise)
                elif perturb_type == 'time_shift':
                    shift = np.random.randint(1, 5)
                    shifted = torch.roll(traj, shifts=shift, dims=0)
                    perturbed.append(shifted)
                else:  # spatial_shift
                    offset = torch.randn(3) * 0.2
                    perturbed.append(traj + offset)
        
        return torch.stack(perturbed)
    
    def generate_random_trajectories(self, num_samples):
        """Generate completely random trajectories."""
        random_trajectories = []
        for _ in range(num_samples):
            traj = torch.randn(50, 3)
            traj = torch.nn.functional.avg_pool1d(
                traj.transpose(0, 1).unsqueeze(0), 
                kernel_size=3, 
                stride=1, 
                padding=1
            ).squeeze(0).transpose(0, 1)
            random_trajectories.append(traj)
        
        return torch.stack(random_trajectories)


class EnhancedEnergyDataset(Dataset):
    """Dataset class for energy-based trajectory learning."""
    
    def __init__(self, good_trajectories, perturbed_trajectories, random_trajectories):
        """
        Initialize the energy dataset.
        
        Args:
            good_trajectories (torch.Tensor): Good trajectories of shape (N, T, 3)
            perturbed_trajectories (torch.Tensor): Perturbed trajectories of shape (N, T, 3)
            random_trajectories (torch.Tensor): Random trajectories of shape (N, T, 3)
        """
        super().__init__()
        
        # Store original trajectories
        self.good_trajectories = good_trajectories
        self.perturbed_trajectories = perturbed_trajectories
        self.random_trajectories = random_trajectories
        
        # Combine all trajectories into a single tensor
        self.trajectories = torch.cat([
            good_trajectories,
            perturbed_trajectories,
            random_trajectories
        ], dim=0)
        
        # Create corresponding energy labels
        self.energies = torch.cat([
            torch.zeros(len(good_trajectories)),
            torch.ones(len(perturbed_trajectories)),
            2 * torch.ones(len(random_trajectories))
        ])
        
        # Compute features for all trajectories
        self.features = self._compute_trajectory_features(self.trajectories)
        
    def _compute_trajectory_features(self, trajectories):
        """
        Compute additional features for each trajectory.
        
        Args:
            trajectories (torch.Tensor): Batch of trajectories of shape (N, T, 3)
            
        Returns:
            torch.Tensor: Computed features for each trajectory
        """
        # Compute velocities (N, T-1, 3)
        velocities = trajectories[:, 1:] - trajectories[:, :-1]
        
        # Compute accelerations (N, T-2, 3)
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        
        # Compute speed (magnitude of velocity) (N, T-1)
        speeds = torch.norm(velocities, dim=2)
        
        # Compute average speed per trajectory (N, 1)
        avg_speeds = speeds.mean(dim=1, keepdim=True)
        
        # Compute total distance traveled (N, 1)
        total_distances = speeds.sum(dim=1, keepdim=True)
        
        # Compute acceleration magnitudes (N, T-2)
        acc_magnitudes = torch.norm(accelerations, dim=2)
        
        # Compute average acceleration (N, 1)
        avg_accelerations = acc_magnitudes.mean(dim=1, keepdim=True)
        
        # Concatenate all features
        features = torch.cat([
            avg_speeds,
            total_distances,
            avg_accelerations
        ], dim=1)
        
        return features
    
    def __len__(self):
        """Return the total number of trajectories in the dataset."""
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """
        Get a single trajectory with its features and energy label.
        
        Args:
            idx (int): Index of the trajectory to retrieve
            
        Returns:
            tuple: (trajectory, features, energy_label)
        """
        # return (
        #     self.trajectories[idx],
        #     self.features[idx],
        #     self.energies[idx]
        # )
        return self.trajectories, self.features, self.energies


if __name__ == "__main__":
    datasets = generate_datasets() # this returns a dict of 3 types of trajectories
