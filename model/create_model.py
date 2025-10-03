from .SRResNet import create_net as load_SRResNet
from .FSRCNN import create_net as load_FSRCNN
from .FSRCNN import FSRCNN
from .ipt import create_net as load_ipt
from .SMFANet import create_net as load_SMFANet
from .SR_MAE import create_net as load_SR_MAE
from .climax import create_net as load_climax
def load_model(model_name):
    if model_name == 'SRResNet':
        model = load_SRResNet()
        return model
    if model_name == 'FSRCNN':
        model = load_FSRCNN()
        return model
    if model_name == 'ipt':
        model = load_ipt()
        return model
    if model_name == 'SMFANet':
        model = load_SMFANet() 
        return model
    if model_name == 'SR_MAE':
        model = load_SR_MAE()
        return model
    if model_name == 'climax':
        model = load_climax()
        return model


        

