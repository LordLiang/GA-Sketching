from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np


class Trainer(object):

    def __init__(self, model, device,
                 train_dataset, val_dataset, exp_name, optimizer='Adam', stage='single'):
        
        self.device = device
        self.model = model.to(self.device)

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.stage = stage
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary')
        self.val_min = None


    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self,batch):
        query = batch.get('points').to(self.device)
        occ = batch.get('occupancies').to(self.device)
        img0 = batch.get('img0').to(self.device)
        calib0 = batch.get('calib0').to(self.device)

        if self.stage == 'single':
            logits = self.model(query, img0, calib0)
        else:
            num_views = np.random.randint(2)+2 # 2,3
            img_lst = [img0]
            calib_lst = [calib0]
        
            img1 = batch.get('img1').to(self.device)
            calib1 = batch.get('calib1').to(self.device)
            img_lst.append(img1)
            calib_lst.append(calib1)

            if num_views > 2: # 3
                img2 = batch.get('img2').to(self.device)
                calib2 = batch.get('calib2').to(self.device)
                img_lst.append(img2)
                calib_lst.append(calib2)

            logits = self.model(query, img_lst, calib_lst, num_views)

        loss = F.binary_cross_entropy_with_logits(logits, occ, reduction='none').sum(-1).mean()

        return loss

    def train_model(self, epochs, num_sample_points):

        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            sum_loss = 0
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            if epoch % 1 == 0:
                self.save_checkpoint(epoch)
                self.model.eval()
                sum_val_loss = 0
                val_data_loader = self.val_dataset.get_loader()
                for batch in val_data_loader:
                    val_loss = self.compute_loss(batch)
                    sum_val_loss += val_loss.item() / num_sample_points

                sum_val_loss = sum_val_loss / len(val_data_loader)

                print('Val loss: ', sum_val_loss)
                
                if self.val_min is None:
                    self.val_min = sum_val_loss

                if sum_val_loss < self.val_min:
                    self.val_min = sum_val_loss
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch),[epoch, sum_val_loss])

                self.writer.add_scalar('val loss', sum_val_loss, epoch)

            for batch in train_data_loader:
                loss = self.train_step(batch)
                loss = loss / num_sample_points
                print("Training loss: ", loss)
                sum_loss += loss

            self.writer.add_scalar('training loss', sum_loss / len(train_data_loader), epoch)

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            if self.stage == 'single':
                torch.save({'epoch':epoch,'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()}, path)
            else:
                torch.save({'epoch':epoch,'model_state_dict': self.model.merger.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)

        if self.stage == 'single':
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.merger.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
