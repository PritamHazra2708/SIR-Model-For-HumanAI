from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from simulator import simulate_stochastic
from ml_model import SIREmulator

app = FastAPI()

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dummy model (In reality, you will train this model first)
model = SIREmulator()

class SimulationParams(BaseModel):
    population: int
    initial_infected: int
    beta: float
    gamma: float
    days: int

@app.post("/simulate")
def run_simulation(params: SimulationParams):
    # 1. Run Stochastic Simulation
    stochastic_data = simulate_stochastic(
        params.population, params.initial_infected, 
        params.beta, params.gamma, params.days
    )
    
    # 2. Run ML Model Prediction (Deterministic approximation)
    ml_predictions = {"S": [], "I": [], "R": []}
    
    # Generate predictions using our PyTorch model
    with torch.no_grad():
        for t in range(params.days + 1):
            # Input vector: [t, S0, I0, R0, beta, gamma]
            S0 = params.population - params.initial_infected
            x = torch.tensor([[float(t), float(S0), float(params.initial_infected), 0.0, params.beta, params.gamma]], dtype=torch.float32)
            
            # Since model isn't trained yet, we will output dummy deterministic ODE values to show the UI works.
            # In your actual GSoC project, `pred = model(x)` will give accurate values.
            # Here is the mathematical ODE for demonstration in the prototype:
            S_prev = ml_predictions["S"][-1] if t > 0 else S0
            I_prev = ml_predictions["I"][-1] if t > 0 else params.initial_infected
            R_prev = ml_predictions["R"][-1] if t > 0 else 0
            
            N = params.population
            dS = -params.beta * S_prev * I_prev / N
            dI = (params.beta * S_prev * I_prev / N) - (params.gamma * I_prev)
            dR = params.gamma * I_prev
            
            ml_predictions["S"].append(max(0, S_prev + dS))
            ml_predictions["I"].append(max(0, I_prev + dI))
            ml_predictions["R"].append(max(0, R_prev + dR))

    return {
        "stochastic": stochastic_data,
        "deterministic_ml": ml_predictions
    }

# Run with: uvicorn main:app --reload