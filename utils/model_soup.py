import torch
import os

def get_model(state_dicts, alpha, model):
    sd = {k : state_dicts[0][k].clone() * alpha for k in state_dicts[0].keys()}
    for i in range(1, len(state_dicts)):
        for k in state_dicts[i].keys():
            sd[k] = sd[k] + state_dicts[i][k].clone() * alpha

    model.load_state_dict(sd)
    return model

def model_load(device, result_path):
    state_dicts=[]
    for f in os.listdir(result_path):
        if f[-2:] == 'pt':
            print(f'Loading {f}')
            state_dicts.append(torch.load(os.path.join(result_path, f), map_location=device))
    return state_dicts