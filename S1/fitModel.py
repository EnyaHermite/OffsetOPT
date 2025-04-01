import numpy as np
import torch, wandb
import torch.nn as nn
from time import time
from datetime import datetime


wandb.init(project="OffsetOPT-StageOne")


class ModelFit(nn.Module):
    
    def __init__(self, model, optimizer, scheduler, loss, device, fout):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.device = device
        self.fout = fout 
        self.decay_steps = 80000
        self._set_template()
        
    def _set_template(self):
        template = "loss: %3.6f, runtime-per-batch: %3.2f ms"
        self.template = {}
        self.template["train"] = "train batches %03d, " + template
        self.template["test"] = "test epoch %03d, " + template
        
    def log_string(self, out_str):
        self.fout.write(out_str+'\n')
        self.fout.flush()
        print(out_str)

    def train_one_epoch(self, epoch, trainLoader):
        start_time = time()
        batch_idx = 0
        
        train_metrics = []
        for batch_data in trainLoader:
            if batch_data is None:
                continue

            batch_data = [item.to(self.device) for item in batch_data]
            points, features, knn_indices, triangles, nv, mf = batch_data
            torch.cuda.empty_cache()

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss, loss_pos_mine, loss_neg_mine = self.loss(logits, points, knn_indices, triangles) 
            loss.backward()
            self.optimizer.step()
            
            batch_idx = batch_idx+1
            last_runtime = (time()-start_time)/batch_idx
                       
            # Gather data and report
            if batch_idx>0 and batch_idx%50==0:
                results = np.asarray(train_metrics).mean(axis=0)
                self.log_string(self.template["train"]%(batch_idx,results[0],last_runtime*1000))
                
            train_metrics.append([loss.item(), loss_pos_mine.item(), loss_neg_mine.item()])
        
        train_metrics = np.asarray(train_metrics)
        loss = np.mean(train_metrics, axis=0)
            
        wandb.log({"train_loss": loss[0], "train_minor_pos": loss[1], "train_minor_neg": loss[2], "lr": self.scheduler.get_last_lr()[0]}, step=epoch)
        return loss[0]    
    
    def test_one_epoch(self, epoch, testLoader):
        batch_idx = 0
        start_time = time()
        
        test_metrics = []
        for batch_data in testLoader:
            if batch_data is None:
                continue

            batch_data = [item.to(self.device) for item in batch_data]
            points, features, knn_indices, triangles, nv, mf = batch_data
            torch.cuda.empty_cache()
            
            logits = self.model(features)
            loss, loss_pos_mine, loss_neg_mine = self.loss(logits, points, knn_indices, triangles) 
            batch_idx = batch_idx+1
        
            test_metrics.append([loss.item(), loss_pos_mine.item(), loss_neg_mine.item()])
            
        runtime = (time()-start_time)/batch_idx
        test_metrics = np.asarray(test_metrics)
        loss = np.mean(test_metrics, axis=0)
        
        self.log_string(self.template["test"]%(epoch,loss[0],runtime*1000))     
        wandb.log({"test_loss": loss[0], "test_minor_pos": loss[1], "test_minor_neg": loss[2]}, step=epoch)
        return loss[0]     
    
    def __call__(self, ckpt_epoch, num_epochs, trainloader, testloader, torch_log_dir):
        self.iters = 0
        for epoch in range(ckpt_epoch, num_epochs):
            self.log_string("************************Epoch %03d Training********************"%(epoch+1))
            self.log_string(str(datetime.now()))
            self.model.train(True)
            self.train_one_epoch(epoch, trainloader)
            if self.scheduler.get_last_lr()[0]>1e-5:
                self.scheduler.step()

            self.log_string("=======================Epoch %03d Evaluation===================="%(epoch+1))
            self.log_string(str(datetime.now()))
            self.model.train(False)
            self.test_one_epoch(epoch, testloader)
            self.log_string("****************************************************************\n")

            # Track best performance, and save the model's state
            if (epoch+1)%10==0:
                model_path = f'{torch_log_dir}/model_{epoch+1}'
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            model_path, _use_new_zipfile_serialization=False)

        # End the wandb run
        wandb.finish()
        return