import re

with open('src/model/straightpcf.py', 'r') as f:
    content = f.read()

replacement = """        cvm_ckpt = cfg.get('cvm_ckpt', None)
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
                    velocity_nets_state_dict[k] = v
                    
            self.velocity_nets.load_parameters(velocity_nets_state_dict)"""

content = re.sub(
    r"        cvm_ckpt = cfg.get\('cvm_ckpt', None\).*?pass", 
    replacement, 
    content, 
    flags=re.DOTALL
)

with open('src/model/straightpcf.py', 'w') as f:
    f.write(content)

