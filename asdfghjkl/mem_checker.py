import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import asdfghjkl as asdl
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG
from asdfghjkl.kernel import batch
from asdfghjkl.precondition import Shampoo,ShampooHyperParams

OPTIM_SGD = 'sgd'
OPTIM_ADAM = 'adamw'
OPTIM_KFAC_MC = 'kfac_mc'
OPTIM_KFAC_EMP = 'kfac_emp'
OPTIM_SKFAC_MC = 'skfac_mc'
OPTIM_SKFAC_EMP= 'skfac_emp'
OPTIM_SMW_NGD = 'smw_ng'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'psgd'
OPTIM_NEWTON = 'newton'
OPTIM_ABS_NEWTON = 'abs_newton'
OPTiM_KBFGS = 'kbfgs'
OPTIM_SHAMPOO='shampoo'

class MemChecker():
    def __init__(self, model: nn.Module,data_set,device):
        self.model = model.to(device)
        self.data_set=data_set
        self.device=device

    def memory_check(self,optim,bslist):
        avail_bs=[]
        for batch_size in bslist:
            max_memory=self.train_two_step(optim,batch_size)
            if max_memory==-1:
                print('batch size',str(batch_size),' is too big')
                break
            else:
                avail_bs.append(batch_size)

        return avail_bs

    def train_two_step(self,opt,batch_size):
        print(str(opt)+' memory checking')
        print('batch size : ',batch_size)
        self.model.train()
        model = self.model
        self.init_all(model, torch.nn.init.normal_, mean=0., std=1)
        train_kwargs = {'batch_size': batch_size, 'drop_last': True}
        train_loader = torch.utils.data.DataLoader(self.data_set, **train_kwargs)

        if opt == OPTIM_ADAM:
            optimizer = optim.AdamW(model.parameters(), lr=0.001,weight_decay=1e-4)
        elif opt == OPTIM_SHAMPOO:
            config = ShampooHyperParams(nesterov=False,weight_decay=1e-4)
            optimizer = Shampoo(model.parameters(),lr=1e-2)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.9,weight_decay=0.01)

        if opt == OPTIM_KFAC_MC:
            config = asdl.NaturalGradientConfig(data_size=batch_size,
                                                fisher_type=FISHER_MC,
                                                ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm],
                                                damping=3)
            grad_maker = asdl.KfacGradientMaker(model, config,swift=False)
        elif opt == OPTIM_KFAC_EMP:
            config = asdl.NaturalGradientConfig(data_size=batch_size,
                                                fisher_type=FISHER_EMP,
                                                ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm],
                                                damping=1)
            grad_maker = asdl.KfacGradientMaker(model, config,swift=False)
        elif opt == OPTIM_SKFAC_MC:
            config = asdl.NaturalGradientConfig(data_size=batch_size,
                                                fisher_type=FISHER_MC,
                                                damping=1,
                                                ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm])
            grad_maker = asdl.KfacGradientMaker(model, config,swift=True)
        elif opt == OPTIM_SKFAC_EMP:
            config = asdl.NaturalGradientConfig(data_size=batch_size,
                                                fisher_type=FISHER_EMP,
                                                damping=0.01,
                                                ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm])
            grad_maker = asdl.KfacGradientMaker(model, config,swift=True)
        elif opt == OPTIM_SMW_NGD:
            config = asdl.SmwEmpNaturalGradientConfig(data_size=batch_size)
            grad_maker = asdl.SmwEmpNaturalGradientMaker(model, config)
        elif opt == OPTIM_KRON_PSGD:
            config = asdl.PsgdGradientConfig()
            grad_maker = asdl.KronPsgdGradientMaker(model,config)
        else:
            grad_maker = asdl.GradientMaker(model)

        try:
            for batch_idx, (x, t) in enumerate(train_loader):
                x, t = x.to(self.device), t.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                dummy_y = grad_maker.setup_model_call(self.model, x)
                grad_maker.setup_loss_call(F.cross_entropy, dummy_y, t)

                y, loss =grad_maker.forward_and_backward()
                optimizer.step()
                
                if batch_idx == 1:
                    break

            del y,loss,optimizer,x,t,grad_maker
            torch.cuda.empty_cache()
            max_memory = torch.cuda.max_memory_allocated()
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print(err)
                max_memory = -1  # OOM

                for param in model.parameters():
                    param.requires_grad = True
                
                del dummy_y,grad_maker,optimizer
                torch.cuda.empty_cache()
            else:
                raise RuntimeError(err)

        return max_memory

    def init_all(self,model, init_func, *params, **kwargs):
        for p in model.parameters():
            init_func(p, *params, **kwargs)