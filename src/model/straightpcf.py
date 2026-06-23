from math import ceil
from typing import Dict, List

import jittor as jt
from jittor import nn
import numpy as np

from .vm import VelocityModule, patch_based_denoise
from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec
from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

class StraightPCFArch(ModelSpec):
    
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
        
        self.velocity_nets = nn.ModuleList()
        for i in range(self.num_modules):
            vm_cfg = cfg.copy()
            if 'num_modules' in vm_cfg:
                del vm_cfg['num_modules']
            if 'cvm_ckpt' in vm_cfg:
                del vm_cfg['cvm_ckpt']
            if 'tot_its' in vm_cfg:
                del vm_cfg['tot_its']
            if 'distance_estimation' in vm_cfg:
                del vm_cfg['distance_estimation']
            
            # Use CVM's feature embedding dimension if provided. 
            # Default to 256 because that is the feature embedding dimension used during the pre-training of CVM models.
            if cfg.get('cvm_ckpt', None) is not None:
                vm_cfg['feat_embedding_dim'] = cfg.get('cvm_feat_embedding_dim', 256)
                
            vm = VelocityModule(model_config=vm_cfg, transform_config=transform_config)
            self.velocity_nets.append(vm)
            
        cvm_ckpt = cfg.get('cvm_ckpt', None)
        if cvm_ckpt is not None:
            # Load pretrained CVM weights
            print(f"Loading pretrained CVM weights from {cvm_ckpt}")
            cvm_state_dict = jt.load(cvm_ckpt)
            
            # The cvm_ckpt is a checkpoint of CoupledVMArch. 
            # It contains keys like 'velocity_nets.0.encoder.conv1.weight'
            # We want to load those into our straightPCF's self.velocity_nets
            # StraightPCF also has self.velocity_nets.
            
            # Filter and construct the state dict for self.velocity_nets
            velocity_nets_state_dict = {}
            for k, v in cvm_state_dict.items():
                if k.startswith('velocity_nets.'):
                    velocity_nets_state_dict[k[len('velocity_nets.'):]] = v
                    
            self.velocity_nets.load_parameters(velocity_nets_state_dict)
                
        self.encoder = FeatureExtraction(k=self.frame_knn, 
                                         input_dim=3, 
                                         embedding_dim=cfg['feat_embedding_dim'], 
                                         distance_estimation=cfg.get('distance_estimation', False))
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3, 
            out_dim=1,
            hidden_size=cfg['decoder_hidden_dim'],
        )

    def parameters(self):
        """
        Return the parameters of the model.
        We exclude the parameters of self.velocity_nets to keep them frozen during training.
        """
        for p in self.encoder.parameters():
            yield p
        for p in self.decoder.parameters():
            yield p
        
    def get_supervised_loss(self, pc_clean, pc_noisy, pc_seeds_t, original_time_step):
        # Ensure velocity_nets stay in eval mode to prevent BatchNorm stats from updating
        self.velocity_nets.eval()
            
        B, N, d = pc_noisy.shape
        pnt_idx = get_random_indices(N, self.num_train_points)
        
        curr_step = original_time_step.unsqueeze(1).unsqueeze(2)
        pc_noisy_interp = curr_step * pc_clean + (1 - curr_step) * pc_noisy

        num = jt.sqrt(((pc_clean - pc_noisy_interp) ** 2).sum(dim=-1))
        den = jt.sqrt(((pc_clean - pc_noisy) ** 2).sum(dim=-1))
        ratio = num[:, 0] / (den[:, 0] + 1e-8)

        # Gather
        pc_clean_gather = pc_clean[:, pnt_idx, :]
        pc_noisy_gather = pc_noisy_interp[:, pnt_idx, :]
        
        feat_d = self.encoder(pc_noisy_gather)
        F_d = feat_d.shape[2]
        pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=len(pnt_idx)).reshape(B) 
        
        loss = ((pred_d - ratio) ** 2).mean()

        for mod in range(self.num_modules):
            with jt.no_grad():
                feat = self.velocity_nets[mod].encoder(pc_noisy_gather)
                F_dim = feat.shape[2]
                pred_dir = self.velocity_nets[mod].decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, len(pnt_idx), d) 
            
            pc_noisy_gather = pc_noisy_gather + (1.0 / self.num_modules) * pred_d.reshape(B, 1, 1) * pred_dir

        finetune_loss = 2e2 * ((pc_clean_gather - pc_noisy_gather) ** 2).sum(dim=-1).mean()

        return (loss + finetune_loss) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=1):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()

            feat_d = self.encoder(pcl_next)
            F_d = feat_d.shape[2]
            pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, 1, 1) 

            for it in range(self.tot_its):
                pred_disp = jt.zeros((B, N, d))
                for mod in range(self.num_modules):
                    feat = self.velocity_nets[mod].encoder(pcl_next)
                    F_dim = feat.shape[2]

                    pred_dir = self.velocity_nets[mod].decoder(
                        c=feat.reshape(-1, F_dim)
                    ).reshape(B, N, d) 
                    
                    pred_disp = (1.0 / self.tot_its) * (1.0 / self.num_modules) * pred_d * pred_dir
                    pcl_next = pcl_next + pred_disp
                        
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        
        if 'time_step' not in batch:
            time_step = jt.rand(pc_noisy.shape[0])
        else:
            time_step = batch['time_step'].reshape(-1)
            
        loss = self.get_supervised_loss(
            pc_clean=pc_clean,
            pc_noisy=pc_noisy,
            pc_seeds_t=jt.zeros_like(pc_clean),
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
                    "pc_noisy": b.meta['pc_noisy'],
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy,
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
