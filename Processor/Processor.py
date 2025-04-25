from __future__ import print_function
import time
import numpy as np
import pickle
from collections import OrderedDict
from tqdm import tqdm
import random
import os
import json
# torch
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from STGCN.feeder import Feeder
from config import CONFIG

def init_seed(_):
    torch.cuda.manual_seed_all(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(name):
    print(f"Attempting to import: {name}")
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Processor():
    def __init__(self):
        self.load_model()
        self.load_data()
        self.load_optimizer()
        self._summary_writer = SummaryWriter(CONFIG.work_dir)
        self.epoch_durations = []


    def load_data(self):
        self.data_loader = dict()
        if CONFIG.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**CONFIG.train_feeder_args),
                batch_size=CONFIG.batch_size,
                shuffle=CONFIG.shuffle,
                num_workers=CONFIG.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**CONFIG.test_feeder_args),
            batch_size=CONFIG.test_batch_size,
            shuffle=False,
            num_workers=CONFIG.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = CONFIG.device[0] if type(CONFIG.device) is list else CONFIG.device
        self.output_device = output_device

        Model = import_class(CONFIG.model)

        self.model = Model(**CONFIG.model_args).cuda(output_device)

        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if CONFIG.weights:
            self.print_log('Load weights from {}.'.format(CONFIG.weights))
            if '.pkl' in CONFIG.weights:
                with open(CONFIG.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(CONFIG.weights, weights_only=True)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in CONFIG.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(CONFIG.device) is list:
            if len(CONFIG.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=CONFIG.device,
                    output_device=output_device)


    def load_optimizer(self):
        #print(' 4 ')
        if CONFIG.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=CONFIG.base_lr,
                momentum=0.9,
                nesterov=CONFIG.nesterov,
                weight_decay=CONFIG.weight_decay)
        elif CONFIG.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=CONFIG.base_lr,
                weight_decay=CONFIG.weight_decay)
        else:
            raise ValueError()

    def adjust_learning_rate(self, epoch):
        if CONFIG.optimizer == 'SGD' or CONFIG.optimizer == 'Adam':
            lr = CONFIG.base_lr * (0.1 ** np.sum(epoch >= np.array(CONFIG.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        if CONFIG.print_log:
            with open('{}/log.txt'.format(CONFIG.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self): 
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def format_time(self,seconds):
        """Convert seconds to hh:mm:ss format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f'{hours:02}:{minutes:02}:{secs:02}'

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        lr = self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        epoch_start_time = time.time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        num_batches = len(loader)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        for batch_idx, (data, label) in enumerate(tqdm(loader, desc="Processing batches")):
            #batch_start_time = time.time()  # Record batch start time
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            output = self.model(data)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #loss_value.append(loss.data[0])  #loss.item()
            loss_value.append(loss.item())

            batch_end_time = time.time()

            #batch_duration = batch_end_time - batch_start_time
            # print(label.cpu().numpy(), output.argmax(dim=1).cpu().numpy())
            accuraciy = accuracy_score(label.cpu().numpy(), output.argmax(dim=1).cpu().numpy())
            precision, recall, f1, _ = precision_recall_fscore_support(label.cpu().numpy(), output.argmax(dim=1).cpu().numpy(),average='weighted')

            accuracies.append(accuraciy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            timer['model'] += self.split_time()
            CONFIG.global_step += 1
            self._summary_writer.add_scalar('iter_loss/train', loss.item(), global_step=CONFIG.global_step)
            if batch_idx % CONFIG.log_interval == 0:
                self.print_log('\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(batch_idx, len(loader), loss.item(), lr))


            timer['statistics'] += self.split_time()

            if (batch_idx + 1) % CONFIG.log_interval == 0:
                elapsed_time = batch_end_time - epoch_start_time
                avg_batch_time = elapsed_time / (batch_idx + 1)
                remaining_batches = num_batches - (batch_idx + 1)
                estimated_remaining_time = avg_batch_time * remaining_batches
                self.print_log(f'Batch {batch_idx + 1}/{num_batches} done. '
                               f'Loss: {loss.item():.4f}. '
                               f'Elapsed Time: {self.format_time(elapsed_time)}s. '
                               f'Estimated Remaining Time: {self.format_time(estimated_remaining_time)}s.')

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        if epoch > 1:
            avg_epoch_time = np.mean(self.epoch_durations)
        else:
            avg_epoch_time = epoch_duration

        self.epoch_durations.append(epoch_duration)
        remaining_epochs = CONFIG.num_epoch - (epoch + 1)
        estimated_remaining_time1 = avg_epoch_time * remaining_epochs

        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        self.print_log('\tEpoch duration: {:.4f} seconds.'.format(epoch_duration))
        self.print_log('\tEstimated Remaining Time: {}'.format(self.format_time(estimated_remaining_time1)))

        self._summary_writer.add_scalar('epoch_loss/train', np.mean(loss_value), global_step=epoch + 1)
        self._summary_writer.add_scalar("train/Accuracies", np.mean(accuracies), global_step=epoch + 1)
        self._summary_writer.add_scalar("train/Precision", np.mean(precisions), global_step=epoch + 1)
        self._summary_writer.add_scalar("train/Recall", np.mean(recalls), global_step=epoch + 1)
        self._summary_writer.add_scalar("train/F1 Score", np.mean(f1_scores), global_step=epoch + 1)

        if save_model:
            model_path = '{}/epoch{}_{}_model.pt'.format(CONFIG.work_dir,epoch + 1, CONFIG.action_type)
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, model_path)


    def eval(self, epoch, save_score=False, loader_name=['test'], save_model=False):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
    
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
    
        for ln in loader_name:
            loss_value = []
            score_frag = []
            for batch_idx, (data, label) in enumerate(self.data_loader[ln]):
                with torch.no_grad():
                    data = data.float().to(self.output_device)
                    label = label.long().to(self.output_device)
    
                output = self.model(data)
    
                loss = self.loss(output, label)
                score_frag.append(output.data.cpu().numpy())
    
                loss_value.append(loss.item())
    
                accuraciy = accuracy_score(label.cpu().numpy(), output.argmax(dim=1).cpu().numpy())
                precision, recall, f1, _ = precision_recall_fscore_support(label.cpu().numpy(), output.argmax(dim=1).cpu().numpy(),average='weighted')
    
                accuracies.append(accuraciy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
    
            score = np.concatenate(score_frag)
            score_dict = dict(zip("sample", score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(ln, len(self.data_loader[ln]), np.mean(loss_value)))
            self._summary_writer.add_scalar('epoch_loss/test', np.mean(loss_value), global_step=epoch + 1)
    
            for k in CONFIG.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
            
            
            self._summary_writer.add_scalar("eval/Accuracies", np.mean(accuracies), global_step=epoch + 1)
            self._summary_writer.add_scalar("eval/Precision", np.mean(precisions), global_step=epoch + 1)
            self._summary_writer.add_scalar("eval/Recall", np.mean(recalls), global_step=epoch + 1)
            self._summary_writer.add_scalar("eval/F1 Score", np.mean(f1_scores), global_step=epoch + 1)
            self.print_log('\tAccuracies: {:.6f}.'.format(np.mean(accuracies)))
            self.print_log('\tPrecision: {:.6f}'.format(np.mean(precisions)))
            self.print_log('\tRecall: {:.6f}.'.format(np.mean(recalls)))
            self.print_log('\tF1 Score: {:.6f}'.format(np.mean(f1_scores)))

            if self.best_acc < np.mean(accuracies) and np.mean(accuracies)>CONFIG.base_acc and save_model:
                model_path = '{}/best_{}_model.pt'.format(CONFIG.work_dir, CONFIG.action_type)
                state_dict = self.model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1],v.cpu()] for k, v in state_dict.items()])
                torch.save(weights, model_path)
                self.best_acc = np.mean(accuracies)
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        CONFIG.work_dir, epoch + 1, ln), 'w') as f:
                    pickle.dump(score_dict, f)
    

    def start(self):
        if CONFIG.phase == 'train':
            self.best_acc =  -1
            self.print_log('Parameters:\n{}\n'.format(str(vars(CONFIG))))
            epoch = CONFIG.start_epoch
            while epoch < CONFIG.num_epoch:
                # for epoch in range(CONFIG.start_epoch, CONFIG.num_epoch):
                save_model = ((epoch + 1) % CONFIG.save_interval == 0) or (epoch + 1 == CONFIG.num_epoch)
                eval_model = ((epoch + 1) % CONFIG.eval_interval == 0) or (epoch + 1 == CONFIG.num_epoch)

                self.train(epoch, save_model=(save_model and CONFIG.save_mode=="regular"))
                if eval_model:
                    self.eval(
                        epoch,
                        save_score=CONFIG.save_score,
                        loader_name=['test'],
                        save_model=(CONFIG.save_mode=="best")
                        )
                else:
                    pass
                epoch += 1
                CONFIG.start_epoch = epoch


        elif CONFIG.phase == 'test':
            if CONFIG.weights is None:
                raise ValueError('Please appoint --weights.')
            CONFIG.print_log = False
            self.print_log('Model:   {}.'.format(CONFIG.model))
            self.print_log('Weights: {}.'.format(CONFIG.weights))

            self.eval(epoch=0, save_score=CONFIG.save_score, loader_name=['test'])

            self.print_log('Done.\n')

            

