from .modules import Model

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam


def validate(model, val_loader, loss_fun, n_samples):

    val_loss = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):

            input, target = batch 

            output = []
            kl_div = []

            for sample in range(n_samples):
                out, kl = model(input)
                output.append(out)
                kl_div.append(kl)

            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)
            log_lik_loss = loss_fun(mean_pred, target)
            loss = log_lik_loss + kl_loss
            val_loss.append(loss)

        mean_loss = np.mean(np.array(val_loss))    

    return mean_loss 




def train(model, optimizer, loss_fun, trainset, valset, device, n_epochs, len_data, n_samples, print_mod = 1):

    loss_lis = []
    overall_loss = []
    val_loss = []

    model = model.to(device)

    N = len_data

    for epoch in range(n_epochs):

        for iter, batch in enumerate(trainset):

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            output = []
            kl_div = []

            for sample in range(n_samples):
                out, kl = model(x)
                output.append(out)
                kl_div.append(kl)

            
            mean_pred = torch.mean(torch.stack(output), dim = 0)
            kl_loss = torch.mean(torch.stack(kl_div), dim = 0)

            loss = loss_fun(mean_pred, y)
            loss += kl_loss #ELBO Loss add if loos_fun is negative log_likelihood

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_lis.append(loss.cpu().detach())

        if epoch % print_mod == 0:

            mean_loss = np.mean(np.array(loss_lis))
            overall_loss += mean_loss
            loss_lis = []

            validation_loss = validate(model, valset, loss_fun, n_samples)

            val_loss.append(validation_loss)


            print(f'Epoch nr {epoch}: mean_train_loss = {mean_loss}, , validation_loss =  {validation_loss}')

    return overall_loss, val_loss




def sample(model, n_samples, testloader):

    model.eval()
    output_list = []
    target_list = []

    with torch.no_grad():

        for data, target in testloader:

            output_mc = []

            for sample in range(n_samples):
                out, _ = model.forward(data)
                output_mc.append(out)

            output = torch.stack(output_mc)
            output_list.append(output)
            target_list.append(target)
        
        mean_pred = torch.mean(torch.stack(output_list), dim = 0)
        std = torch.sqrt(torch.var(torch.stack(output_list), dim = 0))

    return mean_pred, std
        

 # check this    






