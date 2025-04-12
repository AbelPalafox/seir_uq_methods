# -*- coding: utf-8 -*-

import pyro
import pyro.distributions as dist
import torch

from pyro.nn import PyroModule, PyroSample
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm

class pyro_SVI :
    
    def __init__(self, eq_model, guide) :
        
        self.eq_model = eq_model
        self.guide = guide
   
    def infer(self, args, num_iterations, lr=1e-2, **kwargs) :
        # Optimizador Adam
        optimizer = optim.Adam({"lr": lr})
        
        # Definir el objeto de inferencia SVI
        self.svi = SVI(self.eq_model, self.guide, optimizer, loss=Trace_ELBO())
        
        self.losses = []
        # Entrenamiento
        for step in tqdm(range(num_iterations)):
            loss = self.svi.step(args, **kwargs)  # Optimiza los parámetros variacionales
            self.losses.append(loss)
            #if step % 10 == 0:
            #    print(f"Step {step} : loss = {loss:.4f}")