import torch
from torch.nn import DataParallel
from tqdm import tqdm
import sys


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss, loss_weights):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
        self.loss_weights = loss_weights

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss_value = 0
        loss_stats = {key : 0.0 for key in self.loss}
        for key, loss in self.loss.items():
            loss_stats[key] = loss(outputs[-1][key], batch[key])

        loss_value += sum({key: self.loss_weights[key] * val for key, val in loss_stats.items()}.values())
        return outputs[-1], loss_value, loss_stats


class Trainer(object):
    def __init__(self, model, losses, loss_weights,  optimizer=None, num_iter=-1, print_iter=-1, device=None,
                 batches_per_update=1):
        self.num_iter = num_iter
        self.print_iter = print_iter
        self.device = device
        self.batches_per_update = batches_per_update
        self.optimizer = optimizer
        self.loss = losses
        self.loss_weights = loss_weights
        self.model_with_loss = ModelWithLoss(model, self.loss, self.loss_weights)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        results = dict()
        moving_loss = 0
        num_iters = len(data_loader) if self.num_iter < 0 else self.num_iter
        with tqdm(data_loader, file=sys.stdout) as bar_object:
            for iter_id, batch in enumerate(bar_object):

                if iter_id >= num_iters:
                    break
                for k in batch:
                    if k != 'meta':
                        batch[k] = batch[k].to(device=self.device, non_blocking=True)
                if phase == 'train':
                    output, loss, loss_stats = model_with_loss(batch)
                    loss = loss.mean() / self.batches_per_update
                else:
                    with torch.no_grad():
                        output, loss, loss_stats = model_with_loss(batch)
                        loss = loss.mean() / self.batches_per_update
                if phase == 'train':
                    if iter_id % self.batches_per_update == 0:
                        self.optimizer.zero_grad()
                    loss.backward()
                    if (iter_id + 1) % self.batches_per_update == 0:
                        self.optimizer.step()
                moving_loss += loss.item()
                results = {key: results.get(key, 0) + val.mean().item() for (key,val) in loss_stats.items()}
                bar_object.set_postfix_str("{phase}:[{epoch}]' loss={loss} {losses}".
                                           format(phase=phase, epoch=epoch, loss=moving_loss / (iter_id +1),
                                                  losses={k: v / (iter_id +1) for k, v in results.items()}))
                bar_object.update(1)
                if self.print_iter > 0 and not (iter_id % self.print_iter):
                    bar_object.write(bar_object.__str__())
                del output, loss, loss_stats
        results = {k: v / len(data_loader) for k, v in results.items()}
        results.update({'loss': moving_loss / len(data_loader)})
        return results

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)