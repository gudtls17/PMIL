from source.utils import accuracy, TotalMeter, count_params, isfloat, generate_within_brain_negatives, InfoNCE, connectivity_strength_thresholding, get_ITS_wordreport
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
import pickle
from open_clip import get_tokenizer
import time


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.loss_recon = torch.nn.MSELoss(reduction='sum')
        self.loss_cycle = torch.nn.L1Loss(reduction='sum')
        self.loss_clip = InfoNCE(negative_mode='paired', reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph
        self.save_attn_weights = cfg.save_attn_weights
        self.save_test_attn_weights = cfg.save_test_attn_weights
        
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')  # BERT tokenizer
        self.cc200_label_name = np.load("V:/XXX/Area/template/craddock/cc200_label.npy")

        self.init_meters()
        self.mseLoss = torch.nn.MSELoss()
        self.l1Loss = torch.nn.L1Loss()
        
        
        """Freeze text encoder layer"""
        for params in self.model.parameters():
            params.requires_grad = False # all layer Freeze
        for name, params in self.model.named_parameters():
            if 'its_' not in name: # except text encoder layer Unfreeze
                params.requires_grad = True
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        non_trainable_params = total_params - trainable_params
        print("Total params : ", total_params)
        print("Trainable params : ", trainable_params)
        print("Non-trainable params : ", non_trainable_params)
        
        print("Train :", len(self.train_dataloader))
        print("Validation :", len(self.val_dataloader))
        print("Test :", len(self.test_dataloader))

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        for connectivity, timedelay, timescales, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            
            if self.config.preprocess.continus:
                connectivity, timedelay, timescales, label = continus_mixup_data(
                    connectivity, timedelay, timescales, y=label)
            
            connectivity_thr = connectivity_strength_thresholding(connectivity.clone().detach().numpy())
            timescales_text = get_ITS_wordreport(timescales.clone().detach().numpy(), connectivity_thr, self.cc200_label_name)
            
            timescales_text = self.tokenizer(texts=timescales_text, context_length=256)  
            timescales_text = timescales_text.cuda()
            
            connectivity, timedelay, label = connectivity.cuda(), timedelay.cuda(), label.cuda()
            predict, recon_connectivity, recon_timedelay, cycle_connectivity, cycle_timedelay, cls_token  = self.model(connectivity, timedelay, timescales_text) # locket
            
            classifier_loss = self.loss_fn(predict, label)
            recon_loss = self.loss_recon(recon_connectivity, connectivity)/(10*self.config.dataset.node_sz**2) + self.loss_recon(recon_timedelay, timedelay)/(10*self.config.dataset.node_sz**2)
            cycle_loss = self.loss_cycle(cycle_connectivity, connectivity)/(10*self.config.dataset.node_sz**2) + self.loss_cycle(cycle_timedelay, timedelay)/(10*self.config.dataset.node_sz**2)
            
            loss = classifier_loss + recon_loss*0.1 + cycle_loss*0.1 

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for connectivity, timedelay, timescales, label in dataloader:
            
            connectivity_thr = connectivity_strength_thresholding(connectivity.clone().detach().numpy())
            timescales_text = get_ITS_wordreport(timescales.clone().detach().numpy(), connectivity_thr, self.cc200_label_name)
            
            timescales_text = self.tokenizer(texts=timescales_text, context_length=256)  
            timescales_text = timescales_text.cuda()
            
            connectivity, timedelay, label = connectivity.cuda(), timedelay.cuda(), label.cuda()
            output, recon_connectivity, recon_timedelay, cycle_connectivity, cycle_timedelay, cls_token  = self.model(connectivity, timedelay, timescales_text)  # locket

            label = label.float()

            classifier_loss = self.loss_fn(output, label)
            recon_loss = self.loss_recon(recon_connectivity, connectivity)/(10*self.config.dataset.node_sz**2) + self.loss_recon(recon_timedelay, timedelay)/(10*self.config.dataset.node_sz**2)
            cycle_loss = self.loss_cycle(cycle_connectivity, connectivity)/(10*self.config.dataset.node_sz**2) + self.loss_cycle(cycle_timedelay, timedelay)/(10*self.config.dataset.node_sz**2)
            
            loss = classifier_loss + recon_loss*0.1 + cycle_loss*0.1 
            
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def save_attention_weights(self):
        attn_weights = []
        labels = []
        assign_matrices = []
        self.model.eval()
        for connectivity, timedelay, timescales, label in self.train_dataloader:
            
            connectivity_thr = connectivity_strength_thresholding(connectivity.clone().detach().numpy())
            timescales_text = get_ITS_wordreport(timescales.clone().detach().numpy(), connectivity_thr, self.cc200_label_name)
            
            timescales_text = self.tokenizer(texts=timescales_text, context_length=256)  
            timescales_text = timescales_text.cuda()
            
            connectivity, timedelay, label = connectivity.cuda(), timedelay.cuda(), label.cuda()
            prediction, _, _, _, _, _ = self.model(connectivity, timedelay, timescales_text)  # locket

            assignMat = self.model.get_assign_mat()
            assign_np = assignMat.detach().cpu().numpy()
            assign_matrices.append(assign_np)

            attn = self.model.get_attention_weights()
            attn_np = attn[0].detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            labels.append(label_np)
            attn_weights.append(attn_np)


        for connectivity, timedelay, timescales, label in self.val_dataloader:
            
            connectivity_thr = connectivity_strength_thresholding(connectivity.clone().detach().numpy())
            timescales_text = get_ITS_wordreport(timescales.clone().detach().numpy(), connectivity_thr, self.cc200_label_name)
            
            timescales_text = self.tokenizer(texts=timescales_text, context_length=256)  
            timescales_text = timescales_text.cuda()
            
            connectivity, timedelay, label = connectivity.cuda(), timedelay.cuda(), label.cuda()
            prediction, _, _, _, _, _  = self.model(connectivity, timedelay, timescales_text) # locket

            assignMat = self.model.get_assign_mat()
            assign_np = assignMat.detach().cpu().numpy()
            assign_matrices.append(assign_np)


            attn = self.model.get_attention_weights()
            attn_np = attn[0].detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            labels.append(label_np)
            attn_weights.append(attn_np)

        if self.save_test_attn_weights:
            attn_weights_test = []
            labels_test = []
            assign_matrices_test = []     
            
            FCs_test = []
            TDs_test = []
            
            recon_FCs_test = []       
            recon_TDs_test = []
            cycle_FCs_test = []
            cycle_TDs_test = []
            cls_token_test = []

        for connectivity, timedelay, timescales, label in self.test_dataloader:
            
            connectivity_thr = connectivity_strength_thresholding(connectivity.clone().detach().numpy())
            timescales_text = get_ITS_wordreport(timescales.clone().detach().numpy(), connectivity_thr, self.cc200_label_name)
            
            timescales_text = self.tokenizer(texts=timescales_text, context_length=256)  
            timescales_text = timescales_text.cuda()
            
            connectivity, timedelay, label = connectivity.cuda(), timedelay.cuda(), label.cuda()
            prediction, recon_connectivity, recon_timedelay, cycle_connectivity, cycle_timedelay, cls_token = self.model(connectivity, timedelay, timescales_text) # locket

            assignMat = self.model.get_assign_mat()
            assign_np = assignMat.detach().cpu().numpy()
            assign_matrices.append(assign_np)


            attn = self.model.get_attention_weights()
            attn_np = attn[0].detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            labels.append(label_np)
            attn_weights.append(attn_np)

            if self.save_test_attn_weights:
                assign_matrices_test.append(assign_np)
                labels_test.append(label_np)
                attn_weights_test.append(attn_np)
                
                FCs_test.append(connectivity.detach().cpu().numpy())
                TDs_test.append(timedelay.detach().cpu().numpy())
                
                recon_FCs_test.append(recon_connectivity.detach().cpu().numpy())
                recon_TDs_test.append(recon_timedelay.detach().cpu().numpy())
                cycle_FCs_test.append(cycle_connectivity.detach().cpu().numpy())
                cycle_TDs_test.append(cycle_timedelay.detach().cpu().numpy())
                cls_token_test.append(cls_token.detach().cpu().numpy())


        np.save(self.save_path/f"attnWeights.npy", attn_weights, allow_pickle=True)
        np.save(self.save_path/f"labels.npy", labels, allow_pickle=True)
        np.save(self.save_path/f"assign_matrices.npy", assign_matrices, allow_pickle=True)

        if self.save_test_attn_weights:
            np.save(self.save_path/f"attnWeights_test.npy", attn_weights_test, allow_pickle=True)
            np.save(self.save_path/f"labels_test.npy", labels_test, allow_pickle=True)
            np.save(self.save_path/f"assign_matrices_test.npy", assign_matrices_test, allow_pickle=True)
            
            np.save(self.save_path/f"FCs_test.npy", FCs_test, allow_pickle=True)
            np.save(self.save_path/f"TDs_test.npy", TDs_test, allow_pickle=True)
            
            np.save(self.save_path/f"recon_FCs_test.npy", recon_FCs_test, allow_pickle=True)
            np.save(self.save_path/f"recon_TDs_test.npy", recon_TDs_test, allow_pickle=True)
            np.save(self.save_path/f"cycle_FCs_test.npy", cycle_FCs_test, allow_pickle=True)
            np.save(self.save_path/f"cycle_TDs_test.npy", cycle_TDs_test, allow_pickle=True)
            np.save(self.save_path/f"cls_token_test.npy", cls_token_test, allow_pickle=True)

    def generate_save_learnable_matrix(self):

        learable_matrixs = []

        labels = []

        for connectivity, timedelay, timescales, label in self.test_dataloader:
            label = label.long()
            
            connectivity_thr = connectivity_strength_thresholding(connectivity.clone().detach().numpy())
            timescales_text = get_ITS_wordreport(timescales.clone().detach().numpy(), connectivity_thr, self.cc200_label_name)
            
            timescales_text = self.tokenizer(texts=timescales_text, context_length=256)  
            timescales_text = timescales_text.cuda()
            
            connectivity, timedelay, label = connectivity.cuda(), timedelay.cuda(), label.cuda()
            learable_matrix, _, _, _, _, _ = self.model(connectivity, timedelay, timescales_text) # locket

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        best_val_AUC = 0
        best_test_acc = 0
        best_test_AUC = 0
        best_test_sen = 0
        best_test_spec = 0
        best_model = None
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)
            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)
            # raise

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Val Accuracy:{self.val_accuracy.avg: .3f}',
                f'Val Loss{self.val_loss.avg: .3f}',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'Test Spec:{test_result[-2]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.5f}'
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                "Val Loss": self.val_loss.avg,
                "Val Accuracy": self.val_accuracy.avg,
                "Val AUC": val_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
            })

            if(val_result[0] > best_val_AUC):
                best_val_AUC = val_result[0]
                best_test_acc =  self.test_accuracy.avg
                best_test_AUC =  test_result[0]
                best_test_sen = test_result[-1]
                best_test_spec = test_result[-2]
                wandb.run.summary["Best Test Accuracy"] = self.test_accuracy.avg
                wandb.run.summary["Best Test AUC"] = test_result[0]
                wandb.run.summary["Best Val AUC"] = val_result[0]
                wandb.run.summary["Best Val Accuracy"] = self.val_accuracy.avg
                wandb.run.summary["Best Test Sensitivity"] = test_result[-1]
                wandb.run.summary["Best Test Specificity"] = test_result[-2]
                
                best_model = self.model.state_dict()
                if self.save_attn_weights:
                    self.save_attention_weights()

                if self.save_learnable_graph:
                    self.generate_save_learnable_matrix()
                torch.save(best_model, self.save_path/"best_model.pt")
                print("Best Model Updated")
                
            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
                "Val Accuracy": self.val_accuracy.avg,
            })
            

        self.save_result(training_process)
        
        return [best_test_acc,best_test_AUC,best_test_sen,best_test_spec]
