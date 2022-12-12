import torch
import torch.nn as nn
from viphoneme import syms
from tqdm import tqdm



class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def loss_fn(output_phone, phones, phone_lengths):
    fn = nn.CTCLoss(blank=len(syms)+1, zero_infinity=True)
    input_lengths = [output_phone.shape[1] for i in range(output_phone.shape[0])]
    output_phone = output_phone.transpose(0, 1)
    loss = fn(output_phone, phones, input_lengths, phone_lengths)
    return loss


def train_lf(cfg, train_loader, model, optimizer):
    model.train()
    summary_loss = AverageMeter()
    data_len = len(train_loader.dataset) // cfg.batch_size
    with tqdm(total=data_len, leave=True) as pbar:
        for idx, (spectrograms, phones, phone_lengths) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            spectrograms, phones = spectrograms.to(cfg.device), phones.to(cfg.device)
            optimizer.zero_grad()
            output_phone = model(spectrograms)
            loss = loss_fn(output_phone, phones, phone_lengths)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm)
            summary_loss.update(loss.detach().item(), output_phone.shape[0])

            optimizer.step()
            pbar.set_description("Train current loss: {:.4f}".format(loss))
            pbar.update(1)

            # if idx == data_len:
            #     break

    return summary_loss.avg


def val_lf(cfg, val_loader, model):
    model.eval()
    summary_loss = AverageMeter()
    data_len = len(val_loader.dataset) // cfg.batch_size
    with tqdm(total=data_len, leave=True) as pbar:
        for idx, (spectrograms, phones, phone_lengths) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            spectrograms, phones = spectrograms.to(cfg.device), phones.to(cfg.device)
            with torch.no_grad():
                output_phone = model(spectrograms)

            loss = loss_fn(output_phone, phones, phone_lengths)
            summary_loss.update(loss.detach().item(), output_phone.shape[0])
            
            pbar.set_description("Validation current loss: {:.4f}".format(loss))
            pbar.update(1)

            if idx == data_len:
                break
    return summary_loss.avg