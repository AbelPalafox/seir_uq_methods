# -*- coding: utf-8 -*-

import pyro
import pyro.distributions as dist
import torch

from pyro.nn import PyroModule, PyroSample
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO

class pyro_SVI :
    
    def __init__(self, eq_model, **kwargs) :
        
        self.eq_model(**kwargs)
   
    def infer(self, data, num_iterations, lr=1e-2) :
        # Optimizador Adam
        optimizer = optim.Adam({"lr": 0.01})
        
        # Definir el objeto de inferencia SVI
        svi = SVI(self.eq_model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Datos observados (esto debe ser tu conjunto real de datos SEIR)
        observed_data = torch.tensor([data])  # Reemplaza con tus datos reales
        
        # Entrenamiento
        num_iterations = 5000
        for step in range(num_iterations):
            loss = svi.step(observed_data)  # Optimiza los parámetros variacionales
            if step % 500 == 0:
                print(f"Step {step} : loss = {loss:.4f}")