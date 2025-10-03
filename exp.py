import os
import os.path as osp
import json
import torch
import torch.nn as nn
import pickle
import logging
import numpy as np
from model.create_model import load_model
from tqdm import tqdm
from API.metrics import metric
from API.recorder import Recorder
from utils1 import *
from API.dataloader_weatherbench2_sr import load_data
from collections import OrderedDict

def load_model_weights(model, ckpt_path):
    # 加载完整的 checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 创建一个新的状态字典
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k
        if new_key in model.state_dict():  # 检查模型是否有这个层
            # 获取模型中的权重形状
            model_param_shape = model.state_dict()[new_key].shape
            # 检查形状是否匹配
            if model_param_shape == v.shape:
                new_state_dict[new_key] = v
                print(f"Successfully loaded layer {new_key} with shape {model_param_shape}.")
            else:
                print(f"Warning: Shape mismatch for layer {new_key}. Skipping.")
        else:
            print(f"Warning: Layer {new_key} not found in model. Skipping.")

    # 尝试加载匹配的状态字典到模型
    try:
        model.load_state_dict(new_state_dict, strict=False)  # 允许不匹配的层
    except Exception as e:
        print(f"Error loading model weights: {e}")

    # 冻结成功匹配的层（除了名字包含 "filter" 的层）
    for name, param in model.named_parameters():
        if name in new_state_dict and "filter" not in name:
            param.requires_grad = False
            print(f"Frozen layer: {name}")

    return model

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        if 'MAE' in args.model_name:
            model = load_model(args.model_name)
            model = load_model_weights(model,"/home/kongzhibo/SR-MAE/ckpt/checkpoint_repretrain.pth")
            self.model = model.to(self.device)
        else: 
            self.model = load_model(args.model_name).to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        surface_vars = ['u10', 'v10', 't2m']
        pressure_vars = ['t','z','r','u','v']
        pressure_levels = {
        't': [850],
        'z': [500],
        'r': [500],
        'u': [500,850],
        'v': [500,850]
        }
        self.data_mean = np.load("/home/kongzhibo/old_data/mean.npy")
        self.data_std = np.load("/home/kongzhibo/old_data/std.npy")
        self.train_loader, self.vali_loader, self.test_loader = load_data(batch_size=self.args.batch_size,val_batch_size=self.args.val_batch_size,surface_vars=surface_vars,
            pressure_vars=pressure_vars,
            pressure_levels=pressure_levels,
            mean=self.data_mean,
            std=self.data_std,
            pre_len=self.args.pre_len,
            num_workers=4)

    def _select_optimizer(self):
        args = self.args
        if args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.lr, weight_decay=1e-2)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)            
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.lr)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            # Initialize tqdm for training progress
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

            # Track memory usage for each batch
            epoch_memory = []

            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)

                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())

                # Get current GPU memory usage after each batch
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # In MB
                epoch_memory.append(memory_allocated)

                # Update tqdm with the current loss and memory usage
                train_pbar.set_postfix(loss=loss.item(), mem_usage=f"{memory_allocated:.2f} MB")

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)

            # Calculate average memory usage for the epoch
            avg_memory_usage = np.mean(epoch_memory)

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))

                print_log(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} Vali Loss: {vali_loss:.4f} | Avg Memory Usage: {avg_memory_usage:.2f} MB")
                recorder(vali_loss, self.model, self.path)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        value = ['u10', 'v10', 't2m','t850','z500','r500','u500','u850','v500','v850']
        
        for i in range(10):
            rmse, mae, ssim, psnr = metric(pred=preds[:,i,:,:],true=trues[:,i,:,:],mean=vali_loader.dataset.mean[i], std=vali_loader.dataset.std[i], return_ssim_psnr=True)
            print(f"{value[i]} | rmse:{rmse},mae:{mae},ssim:{ssim},psnr:{psnr}")
            print_log('{} | vali rmse:{:.4f},mae:{:.4f},ssim:{:.4f},psnr:{:.4f}'.format(value[i],rmse,mae,ssim,psnr))
        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for i,(batch_x, batch_y) in enumerate(self.test_loader):
            if i * batch_x.shape[0] > 1000:
                break
            pred_y = self.model(batch_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        total_rmse = 0.
        value = ['u10', 'v10', 't2m','t850','z500','r500','u500','u850','v500','v850']
        for i in range(10):
            rmse, mae, ssim, psnr = metric(pred=preds[:,i,:,:],true=trues[:,i,:,:],mean=self.test_loader.dataset.mean[i], std=self.test_loader.dataset.std[i], return_ssim_psnr=True)
            total_rmse = total_rmse+rmse
            print(f"{value[i]} | rmse:{rmse},mae:{mae},ssim:{ssim},psnr:{psnr}")
            print_log('{} | vali rmse:{:.4f},mae:{:.4f},ssim:{:.4f},psnr:{:.4f}'.format(value[i],rmse,mae,ssim,psnr))
        return total_rmse
