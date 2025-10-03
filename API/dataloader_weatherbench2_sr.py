import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset
import xarray as xr 
import os

class WeatherDataset(Dataset):
    def __init__(self, surface_vars, pressure_vars, pressure_levels,mean,std,step=1,is_train=True,pre_len=0):
        super().__init__()
        self.surface_vars = surface_vars
        self.pressure_vars = pressure_vars
        self.pressure_levels = pressure_levels
        self.step = step
        self.mean = mean
        self.std = std
        self.is_train = is_train
        self.pre_len = pre_len
        self.input_surface_files, self.input_pressure_files,self.tar_surface_files,self.tar_pressure_files,self.index = self._load_data()
        self.valid_idx = np.arange(self.index-self.pre_len)

    def _load_data(self):
        # 指定存放 .nc 文件的文件夹路径
        if self.is_train:
            input_surface_data_path = '/home/kongzhibo/old_data/surface_data_1.0_deg'
            input_pressure_data_path = '/home/kongzhibo/old_data/pressure_data_1.0_deg'

            tar_surface_data_path = '/home/kongzhibo/old_data/surface_data'
            tar_pressure_data_path = '/home/kongzhibo/old_data/pressure_data'
        else:
            input_surface_data_path = '/home/kongzhibo/old_data/surface_data_val_1.0_deg'
            input_pressure_data_path = '/home/kongzhibo/old_data/pressure_data_val_1.0_deg'

            tar_surface_data_path = '/home/kongzhibo/old_data/surface_data_val'
            tar_pressure_data_path = '/home/kongzhibo/old_data/pressure_data_val'

        input_surface_files = sorted([os.path.join(input_surface_data_path, f) for f in os.listdir(input_surface_data_path) if f.endswith('.nc')])
        input_pressure_files = sorted([os.path.join(input_pressure_data_path, f) for f in os.listdir(input_pressure_data_path) if f.endswith('.nc')])
        
        tar_surface_files = sorted([os.path.join(tar_surface_data_path, f) for f in os.listdir(tar_surface_data_path) if f.endswith('.nc')])
        tar_pressure_files = sorted([os.path.join(tar_pressure_data_path, f) for f in os.listdir(tar_pressure_data_path) if f.endswith('.nc')])

        index = len(input_surface_files)
        
        return input_surface_files, input_pressure_files, tar_surface_files,tar_pressure_files,index

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        
        input_surface_dataset = xr.open_dataset(self.input_surface_files[index],engine='netcdf4')
        input_pressure_dataset = xr.open_dataset(self.input_pressure_files[index],engine='netcdf4')
        input_surface_dataset = input_surface_dataset.isel(lat=slice(None, None, -1))
        input_pressure_dataset = input_pressure_dataset.isel(lat=slice(None, None, -1))
        
        tar_surface_dataset = xr.open_dataset(self.tar_surface_files[index+self.pre_len],engine='netcdf4')
        tar_pressure_dataset = xr.open_dataset(self.tar_pressure_files[index+self.pre_len],engine='netcdf4')
        
        tar_surface_dataset = tar_surface_dataset.isel(latitude=slice(None, -1))
        tar_pressure_dataset = tar_pressure_dataset.isel(latitude=slice(None, -1))
        
        # surface_dataset = surface_dataset.sel(valid_time=slice(*self.training_time))
        # pressure_dataset = pressure_dataset.sel(valid_time=slice(*self.training_time))
        # 加载表面变量
        input_surface_data_list = []
        tar_surface_data_list = []
        for var in self.surface_vars:
            input_var_data =input_surface_dataset[var].values[np.newaxis, :, :]
            tar_var_data = tar_surface_dataset[var].values[np.newaxis, :, :]
            input_surface_data_list.append(input_var_data)
            tar_surface_data_list.append(tar_var_data)

        # 加载高空变量
        input_pressure_data_list = []
        tar_pressure_data_list = []
        for var in self.pressure_vars:
            for level in self.pressure_levels[var]:
                input_var_data = input_pressure_dataset[var].sel(pressure_level=level).values[np.newaxis, :, :]
                tar_var_data = tar_pressure_dataset[var].sel(pressure_level=level).values[np.newaxis, :, :]
                input_pressure_data_list.append(input_var_data)
                tar_pressure_data_list.append(tar_var_data)
                
        data_inp = np.concatenate(input_surface_data_list + input_pressure_data_list, axis=0)
        data_tar = np.concatenate(tar_surface_data_list + tar_pressure_data_list, axis=0)
        # Normalize data
        if np.any(np.isnan(data_inp)):
            data_inp = np.nan_to_num(data_inp, nan=0)
        if np.any(np.isnan(data_tar)):
            data_tar = np.nan_to_num(data_tar, nan=0)
        data_inp = (data_inp - self.mean) / self.std
        data_tar = (data_tar - self.mean) / self.std
        return data_inp,data_tar

def load_data(batch_size, val_batch_size, surface_vars, pressure_vars, pressure_levels,mean,std,pre_len,num_workers=8, step=1, **kwargs):

    # Create the training set to calculate mean and std
    train_set = WeatherDataset(surface_vars=surface_vars, pressure_vars=pressure_vars, pressure_levels=pressure_levels,step=step,is_train=True,mean=mean,std=std,pre_len=pre_len)
    # Use calculated mean and std from the training set in validation and test sets
    vali_set = WeatherDataset(surface_vars=surface_vars, pressure_vars=pressure_vars, pressure_levels=pressure_levels,mean=mean,std=std,step=step,is_train=False,pre_len=pre_len)
    test_set = WeatherDataset(surface_vars=surface_vars, pressure_vars=pressure_vars, pressure_levels=pressure_levels,mean=mean,std=std,step=step,is_train=False,pre_len=pre_len)
    
    dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_val = torch.utils.data.DataLoader(vali_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    return dataloader_train, dataloader_val, dataloader_test
if __name__ == '__main__':
    surface_vars = ['u10', 'v10', 't2m']
    pressure_vars = ['t','z','r','u','v']
    pressure_levels = {
        't': [850],
        'z': [500],
        'r': [500],
        'u': [500,850],
        'v': [500,850]
        }
    mean = np.load("/home/kongzhibo/data/mean.npy")
    std = np.load("/home/kongzhibo/data/std.npy")
    dataloader_train, dataloader_val, dataloader_test= load_data(
        batch_size=12,
        val_batch_size=12,
        surface_vars=surface_vars,
        pressure_vars=pressure_vars,
        pressure_levels=pressure_levels,
        mean=mean,std=std,
        num_workers=4
    )

    # Testing data loading
    for input_frames, output_frames in iter(dataloader_train):
        np.save(f"/home/kongzhibo/test_data/input.npy", input_frames.numpy())
        np.save(f"/home/kongzhibo/test_data/output.npy", output_frames.numpy())
        break
