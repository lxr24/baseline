import re

with open("src/data/augment.py", "r") as f:
    content = f.read()

content = content.replace(
    "idx = np.random.choice(pc.shape[0], self.num_samples, replace=False)",
    "idx = np.random.permutation(pc.shape[0])[:self.num_samples]"
)
content = content.replace(
    "idx = np.random.choice(pc.shape[0], self.num_samples, replace=True)",
    "idx = np.random.randint(0, pc.shape[0], self.num_samples)"
)

with open("src/data/augment.py", "w") as f:
    f.write(content)
