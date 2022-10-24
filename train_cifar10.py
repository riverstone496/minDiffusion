from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

import argparse,os,sys
import math
import wandb

from asdfghjkl.precondition import Shampoo,ShampooHyperParams
import asdfghjkl as asdl
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG
from asdfghjkl import LOSS_CROSS_ENTROPY,LOSS_MSE

OPTIM_SGD = 'sgd'
OPTIM_ADAM = 'adam'
OPTIM_SHAMPOO='shampoo'
OPTIM_KFAC_MC = 'kfac_mc'
OPTIM_SMW_NGD = 'smw_ngd'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'kron_psgd'
OPTIM_NEWTON = 'newton'
OPTIM_ABS_NEWTON = 'abs_newton'
OPTIM_KBFGS = 'kbfgs'

def train_cifar10(
    n_epoch: int = 100, device: str = "cuda", load_pth: Optional[str] = None
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load("ddpm_cifar.pth"))

    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    test_dataset = CIFAR10(
        "./data",
        train=False,
        download=True,
        transform=tf,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=16)
    #optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    if args.optim == OPTIM_ADAM:
        optim = torch.optim.Adam(ddpm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == OPTIM_SHAMPOO:
        config = ShampooHyperParams(preconditioning_compute_steps=args.interval,statistics_compute_steps=args.interval)
        optim = Shampoo(ddpm.parameters(),lr=args.lr,momentum=args.momentum,hyperparams=config)
    else:
        optim = torch.optim.SGD(ddpm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.optim == OPTIM_KFAC_MC:
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            damping=args.damping,
                                            upd_inv_interval=args.interval,
                                            upd_curvature_interval=args.interval,
                                            loss_type=LOSS_MSE,
                                            ignore_modules=[nn.ConvTranspose2d])
        grad_maker = asdl.KfacGradientMaker(ddpm, config)
    elif args.optim == OPTIM_SMW_NGD:
        config = asdl.SmwEmpNaturalGradientConfig(data_size=args.batch_size,
                                                  damping=args.damping)
        grad_maker = asdl.SmwEmpNaturalGradientMaker(ddpm, config)
    elif args.optim == OPTIM_FULL_PSGD:
        config = asdl.PsgdGradientConfig(upd_precond_interval=args.interval)
        grad_maker = asdl.PsgdGradientMaker(ddpm,config)
    elif args.optim == OPTIM_KRON_PSGD:
        config = asdl.PsgdGradientConfig(upd_precond_interval=args.interval)
        grad_maker = asdl.KronPsgdGradientMaker(ddpm,config)
    elif args.optim == OPTIM_NEWTON:
        config = asdl.NewtonGradientConfig(damping=args.damping)
        grad_maker = asdl.NewtonGradientMaker(ddpm, config)
    elif args.optim == OPTIM_ABS_NEWTON:
        config = asdl.NewtonGradientConfig(damping=args.damping, absolute=True)
        grad_maker = asdl.NewtonGradientMaker(ddpm, config)
    elif args.optim == OPTIM_KBFGS:
        config = asdl.KronBfgsGradientConfig(data_size=args.batch_size,
                                             damping=args.damping)
        grad_maker = asdl.KronBfgsGradientMaker(ddpm, config)
    else:
        grad_maker = asdl.GradientMaker(ddpm)

    if args.lr_scheduler == 'cosine':
        scheduler=CosineAnnealingLR(optim, T_max=args.epochs,eta_min=args.lr*args.lr_ratio)

    config = vars(args).copy()
    config.pop('wandb')
    if args.optim in [OPTIM_SGD, OPTIM_ADAM]:
        config.pop('damping')
    if args.wandb:
        wandb.init(config=config,
                   entity=os.environ.get('WANDB_ENTITY', None),
                   project=os.environ.get('WANDB_PROJECT', None),
                   )

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(train_dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)

            #loss = ddpm(x)
            #loss.backward()

            dummy_y = grad_maker.setup_model_call(ddpm, x)
            grad_maker.setup_loss_repr(dummy_y)
            loss = grad_maker.forward_and_backward()
            loss = ddpm(x)

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), args.kl_clip)
            optim.step()

            if args.wandb:
                log = {'epoch': i,
                       'train_loss': float(loss),
                       'learning_rate': optim.param_groups[0]['lr']}
                wandb.log(log)

            if math.isnan(loss):
                print('Error: Train loss is nan', file=sys.stderr)
                sys.exit(0)

        ddpm.eval()

        test_loss = 0
        with torch.no_grad():
            pbar = tqdm(test_dataloader)
            loss_ema = None
            for x, _ in pbar:
                optim.zero_grad()
                x = x.to(device)
                test_loss += ddpm(x)
                
        test_loss /= len(test_dataloader.dataset)
        pbar.set_description(f"loss: {test_loss:.4f}")
        if args.wandb:
            log = {'epoch': i,
                   'test_loss': test_loss}
            wandb.log(log)
            print(log)

        with torch.no_grad():
            xh = ddpm.sample(8, (3, 32, 32), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/{args.optim}/ddpm_sample_cifar{i}_bs{args.batch_size}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_cifar.pth")

        scheduler.step()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=512,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_ratio', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='learning rate scheduler')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--interval', type=int, default=10,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optim', default=OPTIM_KFAC_MC)
    parser.add_argument('--damping', type=float, default=1e-6)
    parser.add_argument('--kl_clip', type=float, default=1,
                        help='kl_clip')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb', action='store_true', default=False)

    args = parser.parse_args()
    assert torch.cuda.is_available()
    print(args)
    train_cifar10(n_epoch=args.epochs)
