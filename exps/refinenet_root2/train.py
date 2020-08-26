import os
import os.path as osp

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset.p2p_dataset import P2PDataset
from model.refinenet import RefineNet
from config import cfg

checkpoint_dir = cfg.CHECKPOINT_DIR
os.makedirs(checkpoint_dir, exist_ok=True)


def main():
    train_dataset = P2PDataset(dataset_path=cfg.DATA_DIR, root_idx=cfg.DATASET.ROOT_IDX)
    train_loader = DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True)
    
    model = RefineNet()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if len(cfg.MODEL.GPU_IDS) > 1:
        model = nn.parallel.DataParallel(model, device_ids=cfg.MODEL.GPU_IDS)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.LR_STEP_SIZE, gamma=cfg.SOLVER.GAMMA, last_epoch=-1)
    
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS+1):
        total_loss = 0
        count = 0
        for i, (inp, gt) in enumerate(train_loader):
            count += 1
            inp = inp.to(device)
            gt = gt.to(device)

            preds = model(inp)
            loss = criterion(preds, gt)
            total_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        avg_loss = total_loss / count
        if epoch % cfg.PRINT_FREQ == 0:
            print("epoch: {} | loss: {}.".format(epoch, avg_loss))
        if epoch % cfg.SAVE_FREQ == 0 or epoch == cfg.SOLVER.NUM_EPOCHS:
            torch.save(model.state_dict(), osp.join(checkpoint_dir, "RefineNet_epoch_%03d.pth" % epoch))


if __name__ == "__main__":
    main()
