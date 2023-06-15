from tsmixer.data.loader import get_data
from tsmixer.model import TSLinear, TSMixer
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
from argparse import ArgumentParser

device = 'cuda'

def validate(model, valid_loader, criterion):
    loss = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            pred_y = model(batch_x)
            loss.append(criterion(pred_y, batch_y).item())
    return np.average(loss)

def run_exp(args):
    trainset, train_loader = get_data(args.L, args.T, name=args.name, flag='train')
    validset, valid_loader = get_data(args.L, args.T, name=args.name, flag='val')
    testset, test_loader = get_data(args.L, args.T, name=args.name, flag='test')

    if args.model == 'linear':
        model = TSLinear(args.L, args.T).to(device)
    elif args.model == 'tsmixer':
        model = TSMixer(args.L, trainset.C, args.T, args.n_mixer, args.dropout).to(device)
    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(100)):
        total_loss = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            pred_y = model(batch_x)
            
            optimizer.zero_grad()
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.detach().cpu())

        if (epoch + 1) % 5 == 0:
            print("train MSE", np.average(total_loss))
            print("val MSE", validate(model, valid_loader, criterion))
            print("test MSE", validate(model, test_loader, criterion))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-L', type=int, default=512)
    parser.add_argument('-T', type=int, default=96)
    parser.add_argument('--name', type=str, default='ETTh1')
    parser.add_argument('--n_mixer', type=int, default=1)
    parser.add_argument('--model', type=str, default='tsmixer')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.9)

    args = parser.parse_args()
    run_exp(args)
