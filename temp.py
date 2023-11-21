import numpy as np

a = np.array([[100,50],[100, 50]])
print(a.shape)
b = np.array([[125,50],[125,50]])

from torchmetrics.functional import mean_absolute_percentage_error as mape
import torch
print(mape(torch.tensor(a), torch.tensor(b)))


# loss = torch.nn.functional.l1_loss(torch.tensor(a), torch.tensor(b), reduction='none')
# print(loss)
# print(loss.numpy().mean(axis=0))


from sklearn.metrics import mean_absolute_percentage_error

print(mean_absolute_percentage_error(b, a, multioutput="raw_values"))