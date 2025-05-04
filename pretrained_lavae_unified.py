import argparse
import numpy as np
import os
import random
import torch
from model.pretrained import vqvae
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from datafactory.dataloader import loader_provider

def seed_everything(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"seed: {seed_value}")


def plot_comparison(real, reconstructed, save_path):
    for i in range(len(real)):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(real[i], label='Real')
        axs[0].set_title(f'Real Sample {i}')
        axs[1].plot(reconstructed[i], label='Reconstructed')
        axs[1].set_title(f'Reconstructed Sample {i}')
        axs[0].legend()
        axs[1].legend()
        plt.savefig(f"{save_path}/comparison_{i}.png")
        plt.close()


def plot_pca_tsne(real_samples, reconstructed_samples, save_path):
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=min(len(real_samples), 30))
    combined_samples = np.vstack((real_samples, reconstructed_samples))
    combined_samples_pca = pca.fit_transform(combined_samples)
    combined_samples_tsne = tsne.fit_transform(combined_samples)
    labels = ['Real'] * len(real_samples) + ['Reconstructed'] * len(reconstructed_samples)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x=combined_samples_pca[:, 0], y=combined_samples_pca[:, 1], hue=labels, ax=axs[0])
    axs[0].set_title('PCA')
    sns.scatterplot(x=combined_samples_tsne[:, 0], y=combined_samples_tsne[:, 1], hue=labels, ax=axs[1])
    axs[1].set_title('t-SNE')
    plt.legend()
    plt.savefig(f"{save_path}/pca_tsne.png")
    plt.close()


def inference(model, test_loader, device, save_dir, num_samples=None):
    model.eval()
    real_samples = []
    reconstructed_samples = []
    z_samples = []

    with torch.no_grad():
        for i, (data1, data2, data3) in enumerate(test_loader):
            data = (data1, data2, data3)
            for j, data in enumerate(data):
                y_text, batch_x, y_text_embedding = data
                if batch_x == None:
                    continue
                if num_samples is not None and i >= num_samples:  # control sample number
                    break
                real_sample = batch_x.float().to(device)
                print(f"Real sample shape: {real_sample.shape}")
                loss, recon_error, reconstructed_sample, z = model.shared_eval(
                    real_sample, None, mode='test')
                print(f"Reconstructed sample shape: {reconstructed_sample.shape}")
                print(f"Loss for batch {i}: {loss.item()}, Reconstruction Error: {recon_error.item()}")
                if reconstructed_sample.dim() == 1:
                    reconstructed_sample = reconstructed_sample.unsqueeze(0)
                if real_sample.dim() == 2:
                    real_sample = real_sample.unsqueeze(0)
                real_np = real_sample.squeeze().cpu().numpy()
                reconstructed_np = reconstructed_sample.squeeze().cpu().numpy()
                plot_comparison(real_np, reconstructed_np, save_dir)
                real_samples.append(real_np)
                reconstructed_samples.append(reconstructed_np)

    _,real_samples,_ = any_length_evaluation(real_samples)    # you can try any length test here: 24,48,96
    _, reconstructed_samples, _ = any_length_evaluation(reconstructed_samples)
    plot_pca_tsne(real_samples, reconstructed_samples, save_dir)
    mae = np.mean(np.abs(real_samples - reconstructed_samples))
    mse = np.mean((real_samples - reconstructed_samples) ** 2)
    rmse = np.sqrt(mse)
    if num_samples is None:
        metrics_file_path = f"{save_dir}/metrics.txt"
        with open(metrics_file_path, "w") as file:
            file.write(f"MAE: {mae}\n")
            file.write(f"RMSE: {rmse}\n")


def any_length_evaluation(real_samples):
    samples_by_length = {24: [], 48: [], 96: []}
    for sample in real_samples:
        length = sample.shape[1]
        samples_by_length[length].append(sample)
    stacked_by_length = {}
    for length, samples in samples_by_length.items():
        stacked_by_length[length] = np.concatenate(samples, axis=0)
    stacked_24 = stacked_by_length[24]
    stacked_48 = stacked_by_length[48]
    stacked_96 = stacked_by_length[96]
    return stacked_24, stacked_48, stacked_96


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='weather', help='dataset name')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_training_updates', type=int, default=2000, help='number of training updates/epochs')
parser.add_argument('--save_path', type=str,default='results/saved_pretrained_models/', help='denoiser model save path')
# Model-specific parameters
parser.add_argument('--general_seed', type=int, default=42, help='seed for random number generation')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for the optimizer')
parser.add_argument('--block_hidden_size', type=int, default=128, help='hidden size of the blocks in the network')
parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers in the model')
parser.add_argument('--res_hidden_size', type=int, default=256, help='hidden size of the residual layers')
parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of the embeddings')
parser.add_argument('--num_embeddings', type=int, default=128, help='number of embeddings in the VQ-VAE')
parser.add_argument('--compression_factor', type=int, default=4, help='compression factor')
parser.add_argument('--commitment_cost', type=float, default=0.25, help='commitment cost used in the loss function')
parser.add_argument('--mix_train', type=bool, default=True, help='whether to use mixture training')
args = parser.parse_args()

if __name__ == '__main__':
    save_folder_name = 'dataset{}_epoch{}'.format(args.dataset_name, args.num_training_updates)
    save_dir = os.path.join(args.save_path, save_folder_name)
    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(args.general_seed)

    model = vqvae(args).to(device)
    optimizer = model.configure_optimizers(lr=args.learning_rate)
    dataset, train_loader = loader_provider(args, period='train')
    print(len(dataset))
    if args.mix_train:
        for epoch in range(int((args.num_training_updates / len(train_loader)) + 0.5)):
            model.train()
            for i, (data1,data2,data3) in enumerate(train_loader):
                train_loss = []
                data = (data1, data2, data3)
                for j,data in enumerate(data):
                    y_text, batch_x, y_text_embedding = data
                    tensor_all_data_in_batch = batch_x.clone().detach().float().to(device)
                    loss, recon_error, x_recon, z = \
                        model.shared_eval(tensor_all_data_in_batch, optimizer, 'train')
                    train_loss.append(loss.item())
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {np.mean(train_loss).item()}")
            if epoch % (args.num_training_updates/10) == 0:
                torch.save(model, os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
                print(f'Saved Model from epoch: {epoch}')
        torch.save(model, os.path.join(save_dir, 'final_model.pth'))
        print("Training complete.")
    else:
        for epoch in range(int((args.num_training_updates / len(train_loader)) + 0.5)):
            train_loss = []
            for i, data in enumerate(train_loader):
                y_text, batch_x, y_text_embedding = data
                tensor_all_data_in_batch = batch_x.clone().detach().float().to(device)
                loss, recon_error, x_recon, z = \
                    model.shared_eval(tensor_all_data_in_batch, optimizer, 'train')
                train_loss.append(loss.item())
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {np.mean(train_loss).item()}")
            if epoch % (args.num_training_updates / 10) == 0:
                torch.save(model, os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
                print(f'Saved Model from epoch: {epoch}')
        torch.save(model, os.path.join(save_dir, 'final_model.pth'))
        print("Training complete.")

    print("Starting inference...")
    model = torch.load(os.path.join(save_dir, 'final_model.pth'), map_location=device)
    dataset, test_loader = loader_provider(args, period='test')    # note: mixture_dataset is used to train VAE
    inference(model, test_loader, device, save_dir, num_samples=None)
    
