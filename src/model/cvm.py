from math import ceil
from typing import Dict, List

import jittor as jt
from jittor import nn
import numpy as np

from .vm import VelocityModule, patch_based_denoise
from .spec import ModelSpec
from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

class CoupledVMArch(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.tot_its = cfg.get('tot_its', 3)
        self.num_train_points = cfg['num_train_points']
        
        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']
        
        # networks
        self.num_modules = cfg.get('num_modules', 2)
        
        # In python list for Jittor ModuleList equivalent
        self.velocity_nets = nn.ModuleList()
        for i in range(self.num_modules):
            vm_cfg = cfg.copy()
            if 'num_modules' in vm_cfg:
                del vm_cfg['num_modules']
            if 'velocity_ckpt' in vm_cfg:
                del vm_cfg['velocity_ckpt']
            if 'tot_its' in vm_cfg:
                del vm_cfg['tot_its']
            vm = VelocityModule(model_config=vm_cfg, transform_config=transform_config)
            self.velocity_nets.append(vm)
            
        velocity_ckpt = cfg.get('velocity_ckpt', None)
        if velocity_ckpt is not None:
            # Load pretrained VM weights to ALL modules
            print(f"Loading pretrained VM weights from {velocity_ckpt}")
            for i in range(self.num_modules):
                self.velocity_nets[i].load(velocity_ckpt)
    
    def get_supervised_loss(self, pc_clean, pc_noisy, pc_seeds_t, original_time_step):
        B, N, d = pc_noisy.shape
        pnt_idx = get_random_indices(N, self.num_train_points)
        
        # gradient target
        grad_target = pc_clean - pc_noisy

        total_dir_loss = 0.0
        total_consistency_loss = 0.0

        curr_step = (original_time_step * (self.num_modules - 0) + 0) / self.num_modules
        curr_step = curr_step.unsqueeze(1).unsqueeze(2)
        
        pc_noisy_interp = curr_step * pc_clean + (1 - curr_step) * pc_noisy
        # pc_seeds_t here in Jittor baseline could just be mean or not used based on data loader
        # we simplify here to align with what the dataset provides. In the original code pc_seeds_t is used for centering.
        # But in Jittor baseline VM, it works on patch level and the dataloader gives normalized patches.
        # So pc_seeds_t is implicitly 0 because patches are already centered.
        # So we just omit pc_seeds_t subtraction here as long as patches are centered.
        
        # gather
        pc_noisy_gather = pc_noisy_interp[:, pnt_idx, :]
        pc_clean_gather = pc_clean[:, pnt_idx, :]
        pc_noisy_base_gather = pc_noisy[:, pnt_idx, :]
        grad_target_gather = grad_target[:, pnt_idx, :]
        
        for mod in range(self.num_modules):
            feat = self.velocity_nets[mod].encoder(pc_noisy_gather)
            F_dim = feat.shape[2]
            
            pred_dir = self.velocity_nets[mod].decoder(
                c=feat.reshape(-1, F_dim)
            ).reshape(B, len(pnt_idx), d)
            
            dir_loss = (((pred_dir - grad_target_gather) ** 2)).sum(dim=-1).mean()
            total_dir_loss += dir_loss
            
            pc_noisy_gather = pc_noisy_gather + ((1. - original_time_step.unsqueeze(1).unsqueeze(2)) / self.num_modules) * pred_dir
            
            if mod < self.num_modules - 1:
                curr_step_plus_1 = (original_time_step * (self.num_modules - (mod + 1)) + (mod + 1)) / self.num_modules
                curr_step_plus_1 = curr_step_plus_1.unsqueeze(1).unsqueeze(2)
                pc_noisy_interpolated = curr_step_plus_1 * pc_clean_gather + (1 - curr_step_plus_1) * pc_noisy_base_gather
                
                consistency_loss = (((pc_noisy_interpolated - pc_noisy_gather) ** 2)).sum(dim=-1).mean()
                total_consistency_loss += consistency_loss
                
        return (total_dir_loss + 10 * total_consistency_loss) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=1):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for it in range(self.tot_its):
                for mod in range(self.num_modules):
                    feat = self.velocity_nets[mod].encoder(pcl_next)
                    F_dim = feat.shape[2]
                    
                    pred_dir = self.velocity_nets[mod].decoder(
                        c=feat.reshape(-1, F_dim)
                    ).reshape(B, N, d)
                    
                    pcl_next = pcl_next + (1.0 / self.tot_its) * (1.0 / self.num_modules) * pred_dir
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        # pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3) # Jittor baseline dataset gives mixed
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        time_step = batch['time_step'].reshape(-1) # Need to add time_step to dataloader
        
        # if time_step not in batch, generate random uniform [0, 1]
        if 'time_step' not in batch:
            time_step = jt.rand(pc_noisy.shape[0])
            
        loss = self.get_supervised_loss(
            pc_clean=pc_clean,
            pc_noisy=pc_noisy,
            pc_seeds_t=jt.zeros_like(pc_clean), # already centered
            original_time_step=time_step,
        )
        return {"loss": loss}
    
    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)
        
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        
        global_steps = 1 
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            for it in range(global_steps):
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=1000,
                    seed_k=6,
                    seed_k_alpha=1,
                    langevin_steps=1
                )
            pc_denoised = pc_next.detach().numpy()
            res.append({"pc_denoised": pc_denoised})
        return res
        
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
