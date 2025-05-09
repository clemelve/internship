import torch
from tqdm import tqdm


def loss_bvae(recon_x, x, mu, logvar, beta):
    bce = torch.nn.BCEWithLogitsLoss(reduction='mean')(recon_x, x)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld.sum(1).mean(0, True)
    return bce + beta * kld, bce, kld


def loss_bvae_mse(recon_x, x, mu, logvar, beta):
    mse = torch.nn.MSELoss(reduction='sum')(recon_x, x)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld.sum(1).mean(0, True)
    return mse + beta * kld, mse, kld


def train_bVAE(model, train_data_loader, val_data_loader, beta=3, nb_epochs=300, loss_function = loss_bvae_mse, device='cuda' if torch.cuda.is_available() else 'cpu', saving_path=None):
    
    """
    Trains a variational autoencoder. Nothing longitudinal.
    The goal here is because an AE just requires image to train, it's easier to train and used already implemented
    techniques like pin_memory in the data loader.

    :args: model: variational autoencoder model to train
    :args: data_loader: DataLoader to load the training data
    :args: nb_epochs: number of epochs for training
    :args: device: device used to do the variational autoencoder training
    """

    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=1e-8)
    model.to(device)
    model.device = device
    losses = []
    best_val_loss = float('inf')

    for epoch in tqdm(range(1, nb_epochs + 1), desc="Training"):
        model.train()
        train_loss = []
        bce_loss = []
        kld_loss = []
        for x in train_data_loader:

            optimizer.zero_grad()
            x = x.to(device)

            mu, logvar, recon_x, _ = model(x)
            loss, bce, kld = loss_function(recon_x, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss.append(loss.item())
            bce_loss.append(bce.item())
            kld_loss.append(kld.item())

        train_mean_loss = sum(train_loss) / len(train_loss)
        bce_mean_loss = sum(bce_loss) / len(bce_loss)
        kld_mean_loss = sum(kld_loss) / len(kld_loss)


        model.eval()
        val_loss = []
        for x in val_data_loader:
            x = x.to(device)
            with torch.no_grad():
                mu, logvar, recon_x, _ = model(x)
                loss, _, _ = loss_function(recon_x, x, mu, logvar, beta)
                val_loss.append(loss.item())

        val_mean_loss = sum(val_loss) / len(val_loss)

        losses.append(val_mean_loss)
        tqdm.write(f"Epoch {epoch}/{nb_epochs}, Train Loss: {train_mean_loss:.4f}, BCE Loss : {bce_mean_loss:.4f}, KLD Loss : {kld_mean_loss:.4f}, Val Loss : {val_mean_loss:.4f}")

        # Save model if validation loss decreased
        if val_mean_loss < best_val_loss:
            best_val_loss = val_mean_loss
            torch.save(model.state_dict(), saving_path)
            tqdm.write(f"Model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}\n")