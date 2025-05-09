import torch
from tqdm import tqdm
from leaspy import AlgorithmSettings, Data, Leaspy
from utils import build_compatible_leaspy_dataframe
import numpy as np

def fit_longitudinal_estimator_on_nn(encodings_df, longitudinal_estimator, longitudinal_estimator_settings):
    try:
        encodings_data = Data.from_dataframe(encodings_df)
        longitudinal_estimator.fit(encodings_data, longitudinal_estimator_settings)
    except:
        print(encodings_df)
        try:
            encodings_df = encodings_df.reset_index()
            encodings_data = Data.from_dataframe(encodings_df)
            longitudinal_estimator.fit(encodings_data, longitudinal_estimator_settings)
        except:
            print("No that's not the solution")

    return longitudinal_estimator, encodings_df


def project_encodings_for_training(data_df, longitudinal_estimator):
    """
    used for training. For a given encodings dataframe takes the first observation encodings for each patient and
    projects it at the time of the other observations.
    """
    # TODO: Check if we fit on first or on all the observations
    # origin_df = get_lowest_time_per_id_preserve_order(data_df.sort_values(['ID', 'TIME']))
    # maybe try a random projection instead of the first one
    
    data = Data.from_dataframe(data_df)
    settings_personalization = AlgorithmSettings('scipy_minimize', use_jacobian=True)
    ip = longitudinal_estimator.personalize(data, settings_personalization)                             # TODO: Maybe change with sampling instead of parameters chosen to maximize likelihood
    reconstruction_dict = {}
    
    for i in range(len(data_df['ID'].unique())):
        subject_id = data_df['ID'].unique()[i]
        timepoints = data_df[data_df['ID'] == subject_id]['TIME'].tolist()
        timepoints.sort()
        reconstruction_dict[subject_id] = timepoints

    reconstruction = longitudinal_estimator.estimate(reconstruction_dict, ip, to_dataframe=True)

    return reconstruction_dict, reconstruction


def loss_bvae_mse(recon_x, x, mu, logvar, beta):
    mse = torch.nn.MSELoss(reduction='sum')(recon_x, x)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld.sum(1).mean(0, True)
    return mse + beta * kld, mse, kld


def longitudinal_loss(encoded, predicted): 
    return torch.norm((encoded - predicted), dim=1).mean()



def train(model, train_dataloader, val_dataloader, longitudinal_estimator, longitudinal_estimator_settings, nb_epochs=100, beta=3, gamma = 0.3, batch_size =32, save_path = "longitudinal_vae2D_noacc.pth", device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Trains a variational autoencoder on a longitudinal dataset. If longitudinal_estimator is not None then the model
    will be trained in order for its encoding to respect the mixed effect model described by the longitudinal_estimator.
    Just like in the paper:

    :args: model: variational autoencoder model to train
    :args: data_loader: DataLoader to load the training data
    :args: latent_representation_size: number of dimension of the encodings
    :args: longitudinal_estimator: longitudinal mixed model to train
    :args: longitudinal_estimator_settings: training setting of the longitudinal model
    :args: encoding_csv_path: encodings for each observation stored in a CSV (then no need to do it at the beginning of
    the training
    :args: nb_epochs: number of epochs for training
    :args: lr: learning rate of the neural network model
    :args: device: device used to do the variational autoencoder training
    """
    model.to(device)
    model.device = device
    best_loss = float('inf')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=1e-5)

    
    for epoch in tqdm(range(nb_epochs), desc="Training"):
        model.train()
        total_loss = []
        total_recon_loss, total_kl_loss, total_alignment_loss = [], [],[]

        ### Fit the longitudinal mixed effect model

        encodings_df = build_compatible_leaspy_dataframe(train_dataloader, model, device)
        longitudinal_estimator, fitted_encodings = fit_longitudinal_estimator_on_nn(encodings_df,longitudinal_estimator, longitudinal_estimator_settings)
        _, predicted_latent_variables = project_encodings_for_training(fitted_encodings,longitudinal_estimator)


        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = data[0].to(device).float()
            bs = x.shape[0]
            mu, logVar, reconstructed, _ = model(x)

            vae_loss, reconstruction_loss, kl_loss = loss_bvae_mse(reconstructed, x, mu, logVar, beta)

            idex = i*batch_size            
            predicted_longitudinal = torch.tensor(predicted_latent_variables.iloc[idex:idex+bs].values, dtype=torch.float32)
            alignment_loss = longitudinal_loss(mu, predicted_longitudinal.to(device))
            loss = vae_loss + gamma * alignment_loss

            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            total_loss.append(loss.item())
            total_recon_loss.append(reconstruction_loss.item())
            total_kl_loss.append(kl_loss.item())
            total_alignment_loss.append(alignment_loss.item())

    
        model.eval()

        total_val_loss = []

        encodings_df = build_compatible_leaspy_dataframe(val_dataloader, model, device)
        _, predicted_latent_variables = project_encodings_for_training(encodings_df,longitudinal_estimator)
  

        for i, data in enumerate(val_dataloader):
            with torch.no_grad():
                x = data[0].to(device).float()
                mu, logVar, reconstructed, _ = model(x)
                vae_loss, reconstruction_loss, kl_loss = loss_bvae_mse(reconstructed, x, mu, logVar, beta)
                bs = x.shape[0]

                idex = i*batch_size            
                predicted_longitudinal = torch.tensor(predicted_latent_variables.iloc[idex:idex+bs].values, dtype=torch.float32)
                alignment_loss = longitudinal_loss(mu, predicted_longitudinal.to(device))

                loss = reconstruction_loss + model.beta * kl_loss + model.gamma * alignment_loss

            total_val_loss.append(loss.item())

        tqdm.write(f"Epoch {epoch}/{nb_epochs}, Train Loss: {np.mean(total_loss):.4f}, BCE Loss : {np.mean(total_recon_loss):.4f}, KLD Loss : {np.mean(total_kl_loss):.4f}, Alignment Loss : {np.mean(total_alignment_loss):.4f} Val Loss : {np.mean(total_val_loss):.4f}")
        
        if np.mean(total_val_loss) < best_loss:
            best_loss = np.mean(total_val_loss)
            torch.save(model.state_dict(), save_path)

