import argparse
import torch
from matplotlib import pyplot as plt
from model.denoiser.mlp import MLP
from model.denoiser.transformer import Transformer
from datafactory.dataloader import loader_provider
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
from matplotlib.animation import FuncAnimation
import os
import numpy as np
import math
import random

def seed_everything(seed, cudnn_deterministic=False):
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True

def infer(args):
    step = args.total_step
    cfg_scale = args.cfg_scale
    generation_save_path_result = args.generation_save_path_result
    usepretrainedvae = args.usepretrainedvae
    device = args.device

    print(f"Inference config::Step: {step}\t CFG Scale: {cfg_scale}\t Use Pretrained VAE: {usepretrainedvae}")
    os.makedirs(generation_save_path_result, exist_ok=True)
    dataset, dataloader = loader_provider(args, period='test')
    print('dataset length:', len(dataloader))
    model_root_path = args.dataset_name.split('_')[0]
    pretrained_model = torch.load(f'results/saved_pretrained_models/dataset{model_root_path}_epoch2000/final_model.pth',
        map_location=torch.device(device))
    pretrained_model.float().to(device).eval()
    model = {'DiT': Transformer, 'MLP': MLP}.get(args.denoiser)
    if model:
        model = model().to(args.device)
    else:
        raise ValueError(f"No denoiser found")
    model.encoder = pretrained_model.encoder
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])
    model.to(device).eval()
    backbone = {'flowmatching': RectifiedFlow(), 'ddpm': DDPM(args.total_step, args.device)}.get(args.backbone)
    if backbone:
        if args.backbone == 'flowmatching':
            rf = backbone
        elif args.backbone == 'ddpm':
            ddpm = backbone
    else:
        raise ValueError(f"No backbone found")

    x_1_list = []
    x_t_list = []
    y_list = []
    x_t_latent_enc_list = []
    x_t_latent_dec_list = []
    x_infer_list = []
    with (torch.no_grad()):
        for batch, data in enumerate(dataloader):
            print(f'Generating {batch}th Batch TS...')

            y, x_1, embedding = data
            x_1 = x_1.float().to(device)
            embedding = embedding.float().to(device)

            x_t, before = model.encoder(x_1)
            x_t_latent_enc = x_t.clone()
            x_t = torch.randn_like(x_t).float().to(device)
            for j in range(step):
                if args.backbone == 'flowmatching':
                    t = torch.round(torch.full((x_t.shape[0],), j * 1.0 / step, device=device) * step) / step
                    pred_uncond = model(input=x_t, t=t, text_input=None)
                    pred_cond = model(input=x_t, t=t, text_input=embedding)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t = rf.euler(x_t, pred, 1.0 / step)
                elif args.backbone == 'ddpm':
                    t = torch.full((x_t.size(0),), math.floor(step-1-j), dtype=torch.long, device=device)
                    pred_uncond = model(input=x_t, t=t, text_input=None)
                    pred_cond = model(input=x_t, t=t, text_input=embedding)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t = ddpm.p_sample(x_t, pred, t)

                if batch == 0:
                    x_t_infer_stat, after = pretrained_model.decoder(x_t, length=x_1.shape[-1])
                    x_t_infer_stat = x_t_infer_stat.detach().cpu().numpy().squeeze()
                    x_infer_list.append(x_t_infer_stat[0])
            x_t_latent_dec = x_t.clone()
            x_t, after = pretrained_model.decoder(x_t, length=x_1.shape[-1])
            if batch == 0:
                x_t_infer_gt, after = pretrained_model.decoder(x_t_latent_enc, length=x_1.shape[-1])
                x_t_infer_gt = x_t_infer_gt.detach().cpu().numpy().squeeze()
                x_infer_list.append(x_t_infer_gt[0])

            x_1 = x_1.detach().cpu().numpy().squeeze()
            x_t = x_t.detach().cpu().numpy().squeeze()
            x_1_list.append(x_1)
            x_t_list.append(x_t)

            x_t_latent_dec = x_t_latent_dec.detach().cpu().numpy().squeeze()
            x_t_latent_enc = x_t_latent_enc.detach().cpu().numpy().squeeze()

            x_t_latent_dec_list.append(x_t_latent_dec)
            x_t_latent_enc_list.append(x_t_latent_enc)

    x_1_array = np.concatenate(x_1_list, axis=0)
    x_t_array = np.concatenate(x_t_list, axis=0)

    x_t_latent_dec_array = np.concatenate(x_t_latent_dec_list, axis=0)
    x_t_latent_enc_array = np.concatenate(x_t_latent_enc_list, axis=0)

    x_1 = x_1_array[:, :, np.newaxis]
    x_t = x_t_array[:, :, np.newaxis]
    np.save(os.path.join(generation_save_path_result, 'x_1.npy'), x_1)
    np.save(os.path.join(generation_save_path_result, 'x_t.npy'), x_t)
    np.save(os.path.join(generation_save_path_result, 'x_t_latent_dec_array.npy'), x_t_latent_dec_array)
    np.save(os.path.join(generation_save_path_result, 'x_t_latent_enc_array.npy'), x_t_latent_enc_array)

    return x_1, x_t, x_t_latent_dec_array, x_t_latent_enc_array, x_infer_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference flow matching model")
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='Denoiser Model save path')
    parser.add_argument('--usepretrainedvae', default=True, help='pretrained vae')
    parser.add_argument('--backbone', type=str, default='flowmatching', help='flowmatching or DDPM or EDM')
    parser.add_argument('--denoiser', type=str, default='DiT', help='DiT or MLP')
    parser.add_argument('--cfg_scale', type=float, default=7, help='CFG Scale')
    parser.add_argument('--total_step', type=int, default=100, help='total step sampled from [0,1]')

    # for inference
    parser.add_argument('--checkpoint_id', type=int, default=19999,help='model id')
    parser.add_argument('--dataset_name', type=str, default='exchangerate_24', help='dataset name')
    parser.add_argument('--run_multi', type=bool, default=True, help='run multi times for CRPS,MAP,MRR,NDCG')
    args = parser.parse_args()
    args.mix_train = False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_root_path = args.dataset_name.split('_')[0]
    args.checkpoint_path = os.path.join(args.save_path, 'checkpoints', '{}_{}_{}'.format(args.backbone, args.denoiser, model_root_path), 'model_{}.pth'.format(args.checkpoint_id))
    args.generation_save_path = os.path.join(args.save_path, 'generation', '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name,args.cfg_scale,args.total_step))

    if args.run_multi:
        # single
        args.generation_save_path_result = os.path.join(args.generation_save_path)
        x_1, x_t, x_t_latent_dec_array, x_t_latent_enc_array, x_infer_list = infer(args)
        for run_index in range(10):
            # multi
            args.generation_save_path_result = os.path.join(args.generation_save_path, f'run_{run_index}')
            x_1, x_t, x_t_latent_dec_array, x_t_latent_enc_array, x_infer_list = infer(args)
    else:
        args.generation_save_path_result = os.path.join(args.generation_save_path)
        x_1, x_t, x_t_latent_dec_array, x_t_latent_enc_array, x_infer_list = infer(args)
        for i in range(10):
            plt.plot(x_1[i], label="ground truth")
            plt.plot(x_t[i], label="generated")
            plt.legend()
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = np.arange(len(x_infer_list[0]))
        line, = ax.plot(x, x_infer_list[0], color="cornflowerblue", lw=3)
        fixed_line, = ax.plot(x, x_infer_list[-1], color="black", lw=2, label='static line')
        ax.set_ylim(-0.2, 1.5)


        def init():
            line.set_ydata([np.nan] * len(x))
            return line, fixed_line


        def update(frame):
            if frame >= 100:
                line.set_ydata(x_infer_list[-2])
            else:
                line.set_ydata(x_infer_list[frame])
            return line, fixed_line


        ani = FuncAnimation(fig, update, init_func=init, frames=150, interval=500, blit=True)
        ani.save(f"animation_{args.backbone}.gif", fps=25, writer="imagemagick")
