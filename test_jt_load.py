import jittor as jt
from jittor import nn

class A(nn.Module):
    def __init__(self):
        self.a = nn.Linear(1, 1)

class B(nn.Module):
    def __init__(self):
        self.a = nn.Linear(1, 1)
        self.b = nn.Linear(1, 1)

model_a = A()
jt.save(model_a.state_dict(), "test_a.pkl")
model_b = B()
model_b.load("test_a.pkl")
print("success")
