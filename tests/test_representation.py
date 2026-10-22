import pytest
import numpy as np
from src.representation.autoencoder import CompositionalAutoencoder

def test_compositional_autoencoder():
    data = np.random.randn(100, 24).astype(np.float32)
    model = CompositionalAutoencoder(input_dim=24, latent_dim=16)
    
    fit_res = model.fit(data, epochs=3, batch_size=16)
    assert "final_loss" in fit_res
    assert fit_res["epochs"] == 3
    
    latents = model.encode(data)
    assert latents.shape == (100, 16)
    assert not np.isnan(latents).any()
    
    errors = model.get_reconstruction_error(data)
    assert len(errors) == 100
    assert (errors >= 0).all()
