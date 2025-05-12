# Subtle Anomaly Detection Using Longitudinal Model


This project is currently being developed in the context of my end-of-study internship at CREATIS Lab in Lyon.

We propose a method that leverages a Variational Autoencoder (VAE) to model disease progression in medical imaging, using a synthetic dataset (e.g., the Starmen dataset https://doi.org/10.5281/zenodo.5081988) as a preliminary training ground where temporal changes, such as left arm movements, are visually interpretable. The core idea is to use the VAE to reduce the high-dimensional image data into a latent space that is approximately Euclidean under certain conditions. This transformation enables complex, non-Euclidean disease trajectories in image space to be represented as linear paths in the latent space. 

## Method

We begin by training a standard 2D VAE to reconstruct input images while learning compact and meaningful latent representations. Once trained, the convolutions are frozen to preserve visual feature extraction, and the model is fine-tuned using a longitudinal loss that encourages temporally smooth and consistent trajectories across patient timepoints. This loss is defined as the approximation error of a fitted Disease Course Mapping (DCM) model, a mixed-effects model capturing both population trends and individual variations in progression. Through this approach, geodesic disease trajectories in the decoded image space correspond to linear progressions in the latent space, enabling more interpretable and statistically tractable modeling.

After successfully training the longitudinal VAE on the synthetic dataset, we are now exploring its application to subtle anomaly detection. More specifically, identifying whether a subject deviates from the normative (healthy) trajectory. Abnormal subjects are stored in the abnormal_subjects directory. Using the synthetic dataset, our method achieves perfect accuracy, and we are now transitioning to evaluation on real medical imaging data.



P.S.: All the pipeline is available in the notebook.ipynb file. Please ensure that the Leaspy package (https://leaspy.readthedocs.io/en/stable/api.html) is correctly installed and configured before running the DCM modeling.
