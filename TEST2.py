from kan import KAN
import torch
import matplotlib.pyplot as plt

i = 2

match i:
    case 1:
        model = KAN(width=[2,5,1], grid=3, k=3, seed=42)
    case 2:
        model = KAN(width=[2,5,1], k=5, basis='cheb', seed=42)
    case 3:
        model = KAN(width=[2,5,1], k=5, basis='rbf', seed=42)



# Дальше всё как обычно
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
from kan import create_dataset
dataset = create_dataset(f, n_var=2)

# FIT
#model.fit(dataset, opt='Adam', lr=1e-2, steps=500, lamb=1e-4)
model.fit(dataset, opt="LBFGS", steps=25, lamb=0.001);

#PLOT
#model.plot()
#plt.show()

model = model.prune()

#model.plot()
#plt.show()

#model.auto_symbolic()

# FIT
#model.fit(dataset, opt='Adam', lr=1e-2, steps=500, lamb=1e-4)
model.fit(dataset, opt="LBFGS", steps=25, lamb=0.001);

#PLOT
#model.plot()
#plt.show()

model = model.prune()

model.plot()
plt.show()