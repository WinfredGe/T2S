import argparse
import os
import torch
from torch.optim import AdamW, lr_scheduler
from datafactory.dataloader import loader_provider
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
from model.denoiser.transformer import Transformer
from model.denoiser.mlp import MLP
import time

def train(args):
    print(f"Training config::\tepoch: {args.epochs}\tsave_path: {args.save_path}\tdevice: {args.device}")
    os.makedirs(args.save_path, exist_ok=True)
    dataset, dataloader = loader_provider(args,period='train')
    model = {'DiT': Transformer, 'MLP': MLP}.get(args.denoiser)
    if model:
        model = model().to(args.device)
    else:
        raise ValueError(f"No denoiser found")

    pretrained_model = torch.load(args.pretrained_model_path, map_location=torch.device(args.device))
    pretrained_model.float().to(args.device)
    backbone = {'flowmatching': RectifiedFlow(), 'ddpm': DDPM(args.total_step, args.device)}.get(args.backbone)
    if backbone:
        pass
    else:
        raise ValueError(f"No backbone found")

    model.encoder = pretrained_model.encoder
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = not args.usepretrainedvae
    print(f"Total learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"VAE learnable parameters: {sum(p.numel() for p in pretrained_model.encoder.parameters() if p.requires_grad)}")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=len(dataloader) * args.epochs)
    loss_list = []
    start_epoch = 0
    # if from checkpoint:
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        loss_list = checkpoint['loss_list']

    ##################################################
    #                   mix train                  #
    ##################################################
    if args.mix_train:
        print("Mix training...")
        for epoch in range(start_epoch, args.epochs):
            for i, (data1, data2, data3) in enumerate(dataloader):
                print(f"Epoch: {epoch}, batch: {i}")
                batch = i
                data = (data1, data2, data3)
                for j, data in enumerate(data):
                    y_text, x_1, y_text_embedding = data  # y_text:encoded text, x_1: Time Series Data, y_text_embedding: embeded text(using OpenAI embedding API)
                    # print(x_1.shape)
                    if x_1 == None:
                        continue
                    y_text_embedding = y_text_embedding.float().to(args.device)
                    x_1 = x_1.float().to(args.device)
                    x_1, before = model.encoder(x_1)  # TS data ==>VAE==> clear TS embedding

                    if args.backbone == 'flowmatching':
                        t = torch.round(torch.rand(x_1.size(0), device=args.device) * args.total_step) / args.total_step
                        x_t, x_0 = backbone.create_flow(x_1, t)  # x_t: dirty TS embedding, x_0：pure noise
                        noise_gt = x_1 - x_0
                    elif args.backbone == 'ddpm':
                        t = torch.floor(torch.rand(x_1.size(0)).to(args.device) * args.total_step).long()
                        noise_gt = torch.randn_like(x_1).float().to(args.device)
                        x_t, n_xt = backbone.q_sample(x_1, t, noise_gt)
                    else:
                        raise ValueError(f"Unsupported backbone type: {args.backbone}")

                    optimizer.zero_grad()
                    decide = torch.rand(1) < 0.3
                    if decide:
                        y_text_embedding = None
                    pred = model(input=x_t, t=t, text_input=y_text_embedding)
                    loss = backbone.loss(pred, noise_gt)
                    loss.backward()
                    loss_list.append(loss.item())
                    optimizer.step()
                    if batch % 100 == 0:
                        print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()}')
                scheduler.step()

            if epoch % 1000 == 0 or epoch == args.epochs - 1:
                print(f'Saving model {epoch} to {args.save_path}...')
                save_dict = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, loss_list=loss_list)
                torch.save(save_dict, os.path.join(args.save_path, f'model_{epoch}.pth'))
    else:
        ##################################################
        #                   split train                  #
        ##################################################
        print("Not mix training...")
        for epoch in range(start_epoch, args.epochs):
            for batch, data in enumerate(dataloader):
                y_text, x_1, y_text_embedding = data  # y_text:encoded text, x_1: Time Series Data, y_text_embedding: embeded text(using OpenAI embedding API)
                y_text_embedding = y_text_embedding.float().to(args.device)
                x_1 = x_1.float().to(args.device)
                x_1,before = model.encoder(x_1)  # TS data ==>VAE==> clear TS embedding

                if args.backbone == 'flowmatching':
                    t = torch.round(torch.rand(x_1.size(0), device=args.device) * args.total_step) / args.total_step
                    x_t, x_0 = backbone.create_flow(x_1, t)  # x_t: dirty TS embedding, x_0：pure noise
                    noise_gt = x_1 - x_0
                elif args.backbone == 'ddpm':
                    t = torch.floor(torch.rand(x_1.size(0)).to(args.device) * args.total_step).long()
                    noise_gt = torch.randn_like(x_1).float().to(args.device)
                    x_t, n_xt = backbone.q_sample(x_1, t, noise_gt)
                else:
                    raise ValueError(f"Unsupported backbone type: {args.backbone}")

                optimizer.zero_grad()
                decide = torch.rand(1) < 0.3  #  for classifier free guidance
                if decide:
                    y_text_embedding = None
                pred = model(input=x_t, t=t, text_input=y_text_embedding)
                loss = backbone.loss(pred, noise_gt)
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()
                if batch % 100 == 0:
                    print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()}')

            scheduler.step()
            if epoch % 1000 == 0 or epoch == args.epochs - 1:
                print(f'Saving model {epoch} to {args.save_path}...')
                save_dict = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                                 loss_list=loss_list)
                torch.save(save_dict, os.path.join(args.save_path, f'model_{epoch}.pth'))

def get_args():
    parser = argparse.ArgumentParser(description="Train T2S model")
    parser.add_argument('--checkpoint_path', type=str, default='./results/denoiser_results/checkpoints/flowmatching_DiT_weather/model_6000.pth', help='checkpoint path')  #
    parser.add_argument('--dataset_name', type=str, default='weather', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=9216, help='batch_size')
    parser.add_argument('--epochs', type=int, default=20000, help='training epochs')
    parser.add_argument('--save_path', type=str, default='./results/denoiser_results', help='denoiser model save path')
    parser.add_argument('--mix_train', type=bool, default=True, help='mixture train or not')
    # model specific
    parser.add_argument('--usepretrainedvae', default=True, help='pretrained vae')
    parser.add_argument('--total_step', type=int, default=100, help='sampling from [0,1]')
    parser.add_argument('--backbone', type=str, default='flowmatching', help='flowmatching or ddpm or edm')
    parser.add_argument('--denoiser',type=str, default='DiT', help='DiT or MLP')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mix_train:
        args.data_length = 0
    pretrained_special_dataset_name = args.dataset_name.split('_')[0]
    args.pretrained_model_path = f'results/saved_pretrained_models/dataset{pretrained_special_dataset_name}_epoch2000/final_model.pth'
    args.save_path = os.path.join(args.save_path, 'checkpoints', '{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name))
    return args

if __name__ == '__main__':
    args = get_args()
    stime = time.time()
    train(args)
    etime = time.time()
    print(etime - stime)
    
