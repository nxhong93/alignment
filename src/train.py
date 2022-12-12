import __init__
import gc
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW, SGD
from zalo_config import modelConfig
from train_dataset import LyricsTrainDataset
from madgrad import MADGRAD
from model import AcousticModel
from scheduler import GradualWarmupSchedulerV2
from engineer import train_lf, val_lf



class Train_process(object):

    def __init__(self, config=modelConfig):
        super(Train_process, self).__init__()
        self.config = config

    def process_data(self, df, fold_idx):
        train_data = df[df[f'fold'] != fold_idx//2+1].reset_index(drop=True)
        val_data = df[df[f'fold'] == fold_idx//2+1].reset_index(drop=True)

        ex1 = np.random.choice(val_data[val_data['sr']==44100].index)
        ex1 = val_data.loc[ex1, ['audioPath', 'fileName']].values
        ex2 = np.random.choice(val_data[val_data['sr']!=48000].index)
        ex2 = val_data.loc[ex2, ['audioPath', 'fileName']].values

        list_audio = [ex1, ex2]

        train_dataset = LyricsTrainDataset(train_data, partition=f'train{fold_idx}', 
                                           **self.config.dataset_config, transformType=self.config.transformType)
        train_loader = DataLoader(train_dataset, batch_size = self.config.batch_size, 
                                  num_workers=self.config.num_workers, collate_fn=train_dataset.collate_fn)
        valid_dataset = LyricsTrainDataset(val_data, partition=f'validation{fold_idx}', 
                                           **self.config.dataset_config, transformType=self.config.transformType)
        valid_loader = DataLoader(valid_dataset, batch_size = self.config.batch_size, 
                                  num_workers=self.config.num_workers, collate_fn=valid_dataset.collate_fn)

        del train_data, val_data, train_dataset, valid_dataset
        gc.collect()

        return train_loader, valid_loader, list_audio



    def fit(self, df):
        os.makedirs(self.config.save_path, exist_ok=True)
        for fold in range(self.config.fold_num):
            if fold%2==1:
                print(50 *'-')
                print(f'Fold_{fold//2}:')
                model = AcousticModel(**self.config.model_params)
                model = model.to(self.config.device)
                if self.config.opt_type=='madgrad':
                    optimizer = MADGRAD(model.parameters(), lr=self.config.lr)
                elif self.config.opt_type=='adamw':
                    optimizer = AdamW(model.parameters(), lr=self.config.lr)
                elif self.config.opt_type=='sgd':
                    optimizer = SGD(model.parameters(), lr=self.config.lr)

                if self.config.scheduler_type=='cosine':
                    scheduler = self.config.SchedulerCosine(optimizer, **self.config.cosine_params)
                elif self.config.scheduler_type=='reduce':
                    scheduler = self.config.SchedulerReduce(optimizer, **self.config.reduce_params)

                if self.config.has_warmup:
                    scheduler = GradualWarmupSchedulerV2(optimizer, after_scheduler=scheduler,
                                                        **self.config.warmup_params)

                train_loader, valid_loader, list_audio = self.process_data(df, fold)

                best_val_loss = np.Inf
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark= True
                torch.cuda.empty_cache()

                list_train_loss, list_val_loss = [], []
                check = 0
                for epoch in range(self.config.n_epochs):
                    train_loss = train_lf(self.config, train_loader, model, optimizer)
                    val_loss = val_lf(self.config, valid_loader, model)
                    print(f'Epoch{epoch}(lr: {optimizer.param_groups[0]["lr"]:.10f}): Train_loss: {train_loss:.7f} | Val_loss: {val_loss:.7f}')
                    list_train_loss.append(train_loss)
                    list_val_loss.append(val_loss)

                    if best_val_loss > val_loss:
                        best_val_loss = val_loss

                        torch.save(model.state_dict(), f'{self.config.save_path}/Fold{fold}.pth')
                        print('Model improved, saving model!')
                        # self.show_cam(model, list_audio)
                        check=0
                    else:
                        print('Model not improved!')
                        check += 1

                    if self.config.validation_scheduler:
                        scheduler.step()
                    if check >= self.config.number_check:
                        print('Stop training!')
                        break

                # fig = go.Figure()
                # loss_df = pd.DataFrame({'epoch': np.arange(len(list_train_loss)),
                #                         'train_loss': list_train_loss,
                #                         'validation_loss': list_val_loss})
                # loss_df['is_min'] = loss_df['validation_loss'] \
                #     .apply(lambda x :1 if x== loss_df['validation_loss'].min() else 0)

                # train_data = go.Scatter(x=loss_df.epoch, y=loss_df.train_loss, mode='lines+markers', name='train loss')
                # val_data = go.Scatter(x=loss_df.epoch, y=loss_df.validation_loss, mode='lines+markers', name='validation loss')
                # layout = go.Layout(title="Train process", width=500, height=500)

                # fig = go.Figure(data=[train_data, val_data], layout=layout)
                # fig.add_annotation(x=loss_df[loss_df.is_min == 1]['epoch'].max(),
                #                 y=loss_df.validation_loss.min(),
                #                 text=f'{loss_df[loss_df.is_min == 1].epoch.max():.0f}: {loss_df.validation_loss.min():.4f}',
                #                 showarrow=True, arrowhead=1)

                # fig.show()
                # torch.cuda.empty_cache()