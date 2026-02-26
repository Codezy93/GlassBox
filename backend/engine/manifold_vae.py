import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd
import numpy as np
from loguru import logger
from data.preprocess import CONTINUOUS_FEATURES, ALL_FEATURES

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class ManifoldEngine:
    def __init__(self, data_path, latent_dim=4):
        logger.info(f"Initializing ManifoldEngine (Beta-VAE) with latent_dim={latent_dim}")
        splits = joblib.load(data_path)
        splits = joblib.load(data_path)
        self.X_train = splits['X_train']
        # The VAE is trained on SCALED inputs (14 dims)
        self.input_dim = len(CONTINUOUS_FEATURES)
        self.latent_dim = latent_dim
        
        # Load scaler
        scaler_path = data_path.replace("raw_splits.pkl", "scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = VAE(self.input_dim, self.latent_dim).to(self.device)
        self.vae_path = data_path.replace("raw_splits.pkl", "manifold_vae.pt")
        
        if os.path.exists(self.vae_path):
            self.vae.load_state_dict(torch.load(self.vae_path, map_location=self.device))
            logger.info("📖 Manifold VAE loaded.")
        else:
            self.train_vae()

    def train_vae(self, epochs=50, batch_size=64):
        logger.info("🏋️ Training Beta-VAE on credit manifold...")
        # Scale only continuous features for VAE training
        X_cont = self.X_train[CONTINUOUS_FEATURES]
        X_scaled = self.scaler.transform(X_cont)
        data_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        optimizer = optim.Adam(self.vae.parameters(), lr=1e-3)
        
        self.vae.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i+batch_size]
                optimizer.zero_grad()
                
                recon, mu, logvar = self.vae(batch)
                
                # Reconstruction loss (MSE)
                recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                # KL Divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Beta=1.0 for standard VAE, can be tuned
                loss = recon_loss + 1.0 * kl_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_tensor):.4f}")
                
        torch.save(self.vae.state_dict(), self.vae_path)
        logger.success("✅ Manifold VAE training complete.")

    def get_latent_coords(self, input_profile):
        """
        Project high-dim profile into latent space.
        """
        self.vae.eval()
        df = pd.DataFrame([input_profile])[ALL_FEATURES]
        X_scaled = self.scaler.transform(df[CONTINUOUS_FEATURES])
        with torch.no_grad():
            mu, _ = self.vae.encode(torch.FloatTensor(X_scaled).to(self.device))
        return mu.cpu().numpy()[0].tolist()

    def get_manifold_projection(self, n_samples=1000):
        """
        Return a 2D projection (UMAP or just first 2 latent dims) 
        of the training set for the frontend.
        """
        # For simplicity, we'll return the first 2 latent dimensions of a sample
        X_sample = self.X_train.sample(n_samples)[CONTINUOUS_FEATURES]
        X_scaled = self.scaler.transform(X_sample)
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(torch.FloatTensor(X_scaled).to(self.device))
        return mu.cpu().numpy()[:, :2].tolist()
