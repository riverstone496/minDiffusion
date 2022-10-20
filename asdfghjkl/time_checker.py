import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import asdfghjkl as asdl
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG
from asdfghjkl.precondition import Shampoo,ShampooHyperParams,PRECONDITIONER,STEP,GRAFT,Preconditioner
import time,math
import csv

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

class TimeChecker():
    def __init__(self, model: nn.Module,data_set,device,criterion=nn.CrossEntropyLoss(),thratio=[2],num_iters=100, num_warmups=5):
        self.model = model.to(device)
        self.data_set=data_set
        self.device=device
        self.criterion = criterion
        self.num_iters= num_iters
        self.num_warmups=num_warmups
        self.datasize=len(data_set)
        self.csvlist=[]

        self.thratio=thratio

        # iter times in Shampoo profile
        self.trial_iter = 1
        
    def add_csv(self,opt,iter,bs,fix_time,change_time,intervals):
        for thratio,interval in intervals.items():
            dic={'opt':opt,'iter':iter,'thratio':thratio,'bs':bs,'fix_time':fix_time,'change_time':change_time,'interval':interval}
            self.csvlist.append(dic)
    
    def out_csv(self,file_name):
        print(self.csvlist[0])
        field_name=self.csvlist[0].keys()
        with open(file_name,'w',encoding='utf-8',newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = field_name)
            writer.writeheader()
            writer.writerows(self.csvlist)

    # measuring time of functions
    def time_funcs(self,funcs, func_names=None, num_iters=100, num_warmups=5):
        if func_names is None:
            func_names = [f.__name__ for f in funcs]
        
        time_dic={}
        time_dic=dict.fromkeys(func_names,0)
        for _ in range(num_warmups):
            for f in funcs:
                f()
        for f, fname in zip(funcs, func_names):
            start = time.time()
            for i in range(num_iters):
                f()
                torch.cuda.synchronize()
                if time.time() - start>30:
                    break

            #average time
            time_dic[fname] = (time.time() - start)/(i+1)

        return time_dic

    def decide_interval(self,batch_size,fix_time,change_time,sgd_time):
        intervals={}
        total_time={}
        for th_ratio in self.thratio:
            if change_time==0:
                if fix_time > th_ratio*sgd_time['sum']:
                    interval=-1
                    step_time=-1
                else:
                    interval=1
                    step_time=fix_time
            else:
                interval = change_time/(th_ratio*sgd_time['sum']-fix_time)
                if interval<0:
                    interval=-1
                    step_time=-1
                interval=math.ceil(interval)
                step_time=fix_time+change_time/interval

                # batch sizeによる上限を設けない．
                #if interval > self.datasize/batch_size:
                #    interval=-1
                #    step_time=-1
            intervals[th_ratio]=interval
            total_time[th_ratio]=step_time*self.datasize/batch_size
        return intervals,total_time

    def opt_throughput(self,optim,batch_size,sgd_time,iter_num=20):
        self.init_all(self.model, torch.nn.init.normal_, mean=0., std=1) 

        if optim == OPTIM_SKFAC_MC:
            kfac_time = self.kfac_mc_throughput(batch_size,fisher_type=FISHER_MC,swift=True)
            # sgd_time['sum'] = = fwd + bwd + upd_param
            fix_time = sgd_time['sum'] + kfac_time['precond']
            # kfac_time['fwd_bwd_curv'] = fwd + bwd + bwd2 + upd_curv
            change_time = kfac_time['fwd_bwd_curv']  - sgd_time['fwd'] - sgd_time['bwd'] -  kfac_time['precond']
        elif optim == OPTIM_SKFAC_EMP:
            kfac_time = self.kfac_mc_throughput(batch_size,fisher_type=FISHER_EMP,swift=True)
            # sgd_time['sum'] = = fwd + bwd + upd_param
            fix_time = sgd_time['sum'] + kfac_time['precond']
            # kfac_time['fwd_bwd_curv'] = fwd + bwd + bwd2 + upd_curv
            change_time = kfac_time['fwd_bwd_curv']  - sgd_time['fwd'] - sgd_time['bwd'] -  kfac_time['precond']
        elif optim == OPTIM_KFAC_MC:
            kfac_time = self.kfac_mc_throughput(batch_size,fisher_type=FISHER_MC,swift=False)
            fix_time = sgd_time['sum'] + kfac_time['precond']
            change_time = kfac_time['fwd_bwd_curv']  - sgd_time['fwd']- sgd_time['bwd'] - kfac_time['precond']
        elif optim == OPTIM_SMW_NGD:
            smw_time = self.smw_ng_throughput(batch_size)
            fix_time = smw_time['fwd_bwd_curv'] + sgd_time['upd_param']
            change_time = 0
        elif optim == OPTIM_KRON_PSGD:
            psgd_time = self.psgd_throughput(batch_size)
            fix_time = sgd_time['sum'] + psgd_time['precondition'] 
            change_time = psgd_time['fwd_bwd_upd_precond'] - sgd_time['fwd'] - sgd_time['bwd'] 
        elif optim == OPTIM_SHAMPOO:
            shampoo_time = self.shampoo_throughput(batch_size)
            fix_time = sgd_time['sum'] + shampoo_time['precondition'] 
            change_time = shampoo_time['inverse']*iter_num/self.trial_iter +shampoo_time['upd_curv']

        return fix_time,change_time

    def sgd_throughput(self,batch_size):
        torch.cuda.empty_cache()
        self.init_all(self.model, torch.nn.init.normal_, mean=0., std=1) 
        self.model.train()
        model = self.model
        train_kwargs = {'batch_size': batch_size, 'drop_last': True}
        train_loader = torch.utils.data.DataLoader(self.data_set, **train_kwargs)

        opt = optim.SGD(model.parameters(), lr=1e-2,momentum=0.9)

        for batch_idx, (x, t) in enumerate(train_loader):
            inputs, labels = x.to(self.device), t.to(self.device)
            break

        outputs = model(inputs)
        loss = self.criterion(outputs, labels)

        def fwd():
            nonlocal loss
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)

        def bwd():
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
    
        def upd_param():
            opt.step()

        sgd_time = self.time_funcs([fwd,bwd,upd_param],num_iters=self.num_iters,num_warmups=self.num_warmups)

        sgd_time['sum'] = sgd_time['fwd']+ sgd_time['bwd']+ sgd_time['upd_param']

        del loss,opt

        return sgd_time
        
    def kfac_mc_throughput(self,batch_size,fisher_type,swift):
        self.model.train()
        model = self.model
        train_kwargs = {'batch_size': batch_size, 'drop_last': True}
        train_loader = torch.utils.data.DataLoader(self.data_set, **train_kwargs)        

        config = asdl.NaturalGradientConfig(data_size=batch_size,
                                            fisher_type=fisher_type,
                                            damping=0.01,
                                            ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm])
        grad_maker = asdl.KfacGradientMaker(model,config,swift=swift)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            break

        dummy_y = grad_maker.setup_model_call(model, inputs)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y, labels)

        def fwd_bwd_curv():
            model.zero_grad(set_to_none=True)
            grad_maker.forward_and_backward(accumulate=False)

        def precond():
            grad_maker.precondition()

        return self.time_funcs([fwd_bwd_curv,precond],num_iters=self.num_iters,num_warmups=self.num_warmups)
        
    def smw_ng_throughput(self,batch_size):
        self.model.train()
        model = self.model
        train_kwargs = {'batch_size': batch_size, 'drop_last': True}
        train_loader = torch.utils.data.DataLoader(self.data_set, **train_kwargs)        

        config = asdl.SmwEmpNaturalGradientConfig(data_size=batch_size,damping=0.1)
        grad_maker = asdl.SmwEmpNaturalGradientMaker(model, config)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            break

        dummy_y = grad_maker.setup_model_call(model, inputs)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y, labels)

        def fwd_bwd_curv():
            model.zero_grad(set_to_none=True)
            grad_maker.forward_and_backward()


        return self.time_funcs([fwd_bwd_curv],num_iters=self.num_iters,num_warmups=self.num_warmups)

    def psgd_throughput(self,batch_size):
        self.model.train()
        model = self.model
        train_kwargs = {'batch_size': batch_size, 'drop_last': True}
        train_loader = torch.utils.data.DataLoader(self.data_set, **train_kwargs)        

        grad_maker = asdl.KronPsgdGradientMaker(model)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            break

        dummy_y = grad_maker.setup_model_call(model, inputs)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y, labels)

        grad_maker.forward()
        loss=grad_maker._loss

        def fwd_bwd_upd_precond():
            y=model(inputs)
            loss=self.criterion(y,labels)
            grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
            for p, g in zip(model.parameters(), grads):
                p.grad = g
            grad_maker.update_preconditioner(retain_graph=False)

        def precondition():
            grad_maker.precondition()

        retdic=self.time_funcs([fwd_bwd_upd_precond,precondition],num_iters=self.num_iters,num_warmups=self.num_warmups)
        
        del loss,grad_maker
        torch.cuda.empty_cache()

        return retdic

    def shampoo_throughput(self,batch_size):
        self.model.train()
        model = self.model
        train_kwargs = {'batch_size': batch_size, 'drop_last': True}
        train_loader = torch.utils.data.DataLoader(self.data_set, **train_kwargs)        

        config = ShampooHyperParams(num_iters=self.trial_iter,error_torelance=0)
        opt = Shampoo(model.parameters(),lr=1e-5,momentum=0.9,hyperparams=config)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            break

        def upd_curv():
            for group in opt.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Shampoo does not support sparse yet')
                    state = opt.state[p]
                    if not state:
                        opt.init_var_state(p, state)
                    state[STEP] += 1

                    preconditioner = state[PRECONDITIONER]
                    graft = state[GRAFT]

                    # Gather statistics, compute preconditioners
                    graft.add_statistics(grad)
                    preconditioner.add_statistics(grad)

        def inverse():
            for group in opt.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    state = opt.state[p]
                    preconditioner = state[PRECONDITIONER]
                    preconditioner.compute_preconditioners(num_iters=self.trial_iter,error_torelance=-1)

        def precondition():
            for group in opt.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Shampoo does not support sparse yet')
                    state = opt.state[p]
                    if not state:
                        opt.init_var_state(p, state)
                    state[STEP] += 1

                    preconditioner = state[PRECONDITIONER]
                    graft = state[GRAFT]

                    graft_grad = graft.precondition_gradient(grad)
                    shampoo_grad = preconditioner.preconditioned_grad(grad)

        return self.time_funcs([upd_curv,inverse,precondition],num_iters=self.num_iters,num_warmups=self.num_warmups)

    def init_all(self,model, init_func, *params, **kwargs):
        for p in model.parameters():
            init_func(p, *params, **kwargs)