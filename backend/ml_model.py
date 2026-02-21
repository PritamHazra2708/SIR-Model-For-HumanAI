import torch
import torch.nn as nn

# Simple Neural Network to predict S, I, R over time
class SIREmulator(nn.Module):
    def __init__(self):
        super(SIREmulator, self).__init__()
        # Inputs: t, S0, I0, R0, beta, gamma (6 inputs)
        # Outputs: S, I, R at time t (3 outputs)
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# Function to demonstrate Auto-Differentiation (Crucial for the proposal)
def compute_derivatives(model, inputs):
    # inputs tensor must have requires_grad=True to compute derivatives wrt time
    inputs.requires_grad_(True)
    
    predictions = model(inputs)
    S_pred, I_pred, R_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    
    # Auto-diff to find dS/dt, dI/dt, dR/dt
    # Assuming time 't' is the first feature in the inputs tensor: inputs[:, 0]
    dS_dt = torch.autograd.grad(S_pred, inputs, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0][:, 0]
    dI_dt = torch.autograd.grad(I_pred, inputs, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0][:, 0]
    dR_dt = torch.autograd.grad(R_pred, inputs, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0][:, 0]
    
    return dS_dt, dI_dt, dR_dt