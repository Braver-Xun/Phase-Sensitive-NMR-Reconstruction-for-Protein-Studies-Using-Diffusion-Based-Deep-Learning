import scipy.io as scio
import numpy as np
def pad_data(data, block_size):
    height, width = data.shape
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    padded_data = np.pad(data, ((0, pad_height), (0, pad_width)), mode='constant')
    return padded_data, pad_height, pad_width

def block_data(data, block_size):
    padded_data, pad_height, pad_width = pad_data(data, block_size)
    height, width = padded_data.shape
    num_blocks_vertical = height // block_size
    num_blocks_horizontal = width // block_size
    blocks = np.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size), dtype=padded_data.dtype)
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = padded_data[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            blocks[idx] = block
            idx += 1
    
    return blocks, padded_data.shape, pad_height, pad_width

def reconstruct_data(blocks, pad_shape, pad_height, pad_width):
    height, width = pad_shape
    num_blocks, block_height, block_width = blocks.shape
    num_blocks_vertical = height // block_height
    num_blocks_horizontal = width // block_width
    reconstructed_data = np.empty((height, width), dtype=blocks.dtype)
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = blocks[idx]
            reconstructed_data[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = block
            idx += 1
    final_data = reconstructed_data[:height-pad_height, :width-pad_width]
    
    return final_data

ori_spec = scio.loadmat("gb3.mat")['LR']
phase_input, pad_shape, pad_height, pad_width = block_data(ori_spec, 256)
GT_label = phase_input
block_num = phase_input.shape[0]

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import torch.cuda as cuda
import scipy.io as scio
import scipy.interpolate
import model as Model
import argparse
import core.logger as Logger
import core.metrics as Metrics
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="config/sr_sr3_256_256_condition.json",
                        help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'], 
                        help='Run either train(training) or val(generation)', default='val')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_wandb_ckpt', action='store_true')
parser.add_argument('-log_eval', action='store_true')

args = parser.parse_args(args=[])
opt = Logger.parse(args)
opt = Logger.dict_to_nonedict(opt)

# logging
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if cuda.is_available() else 'cpu')

diffusion = Model.create_model(opt)

max_amp = np.max(np.abs(phase_input))
Phase_Input = torch.complex(torch.from_numpy(phase_input.real).float(), torch.from_numpy(phase_input.imag).float()) / np.max(np.abs(phase_input))
GT_Label = torch.from_numpy(GT_label).float() / np.max(GT_label)
Abs_Input = torch.from_numpy(np.abs(phase_input)).float() / np.max(np.abs(phase_input))

LR = Abs_Input.unsqueeze(1)
SR = torch.cat((Phase_Input.real.unsqueeze(1), Phase_Input.imag.unsqueeze(1)), dim=1)
HR = GT_Label.unsqueeze(1)
val_data = {'LR': LR, 'HR': HR, 'SR': SR}

diffusion = Model.create_model(opt)
avg_mse = 0.0

diffusion.set_new_noise_schedule(
    opt['model']['beta_schedule']['val'], schedule_phase='val')

val_data = diffusion.set_device(val_data)
diffusion.feed_data(val_data)
diffusion.test(continous=False)
# print(idx)
visuals = diffusion.get_current_visuals()
print(visuals['SR'].shape)
sr_img = reconstruct_data(visuals['SR'][-block_num : ].squeeze(1).float().cpu().numpy(), pad_shape, pad_height, pad_width)
hr_img = reconstruct_data(visuals['HR'][0 : block_num].squeeze(1).float().cpu().numpy(), pad_shape, pad_height, pad_width)
lr_img = reconstruct_data(visuals['LR'][0 : block_num].squeeze(1).float().cpu().numpy(), pad_shape, pad_height, pad_width)
fake_img = ori_spec

fake_img_abs = np.abs(fake_img)
max_fake_img = np.max(fake_img_abs)
min_fake_img = np.min(fake_img_abs)
fake_img = fake_img / max_fake_img

lr_img = lr_img / np.max(lr_img)

sr_img_abs = np.abs(sr_img)
max_sr_img = np.max(sr_img_abs)
min_sr_img = np.min(sr_img_abs)
sr_img = sr_img / max_sr_img

hr_img_abs = np.abs(hr_img)
max_hr_img = np.max(hr_img_abs)
min_hr_img = np.min(hr_img_abs)
hr_img = hr_img / max_hr_img

# generation
avg_mse = Metrics.calculate_mse(sr_img, hr_img)
print(avg_mse)  


scio.savemat("gb3_out.mat", {'recon_data': sr_img, 'ori_data':fake_img})