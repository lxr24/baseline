import yaml

with open('configs/model/straightpcf.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Just check if it's there
print("cvm_ckpt in straightpcf.yaml?", 'cvm_ckpt' in data)

