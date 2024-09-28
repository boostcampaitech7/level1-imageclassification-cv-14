import torch
import math
import os



def get_model(state_dicts, alphal, model):
    alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
    # model = ModelWrapper(model, feature_dim, num_classes, normalize)
    sd = {k : state_dicts[0][k].clone() * alphal[0] for k in state_dicts[0].keys()}
    for i in range(1, len(state_dicts)):
        for k in state_dicts[i].keys():
            sd[k] = sd[k] + state_dicts[i][k].clone() * alphal[i]

    model.load_state_dict(sd)


    return model

def model_load(device):
    state_dicts=[]
    for f in os.listdir('./train_result'):
        print(f)
        if f[-2:] == 'pt':
            print(f'Loading {f}')
            state_dicts.append(torch.load('./train_result/'+f, map_location=device))
    return state_dicts