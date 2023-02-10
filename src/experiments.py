from argparse import ArgumentParser
import data
from models.uplift_mlp import UpliftMLP
from models.uplift_resnet import UpliftResNet

import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')    
    parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='num epochs (default: 1)')  
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='maximum num steps (disabled by default: -1)')                          
    parser.add_argument('--l2', type=float, default=1e-3,
                    help='l2 regularization for each learner')    
    parser.add_argument('--l2_diff', type=float, default=1e-3,
                    help='l2 regularization for both learners')   
    parser.add_argument('--grad_clip', type=float, default=-1,
                    help='gradient clipping (default -1, ignored')  
    parser.add_argument('--lr_scheduler', type=str,
                    help='learning rate scheduler; default none')                     
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')  
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')    
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units per layer (default: 256)')    
    parser.add_argument('--layers', type=int, default=1,
                        help='# of levels (default: 1)')
    parser.add_argument('--layer_norm', type=bool, default=False,
                        help='if we should use layer normalization (default: False)')
    parser.add_argument('--accelerator', type = str, default="cpu",
                        help="accelator to use (default cpu, use 'gpu'")
    parser.add_argument('--name', type=str, default="my_model",
                        help='name used for logging (default: my_model)')
    parser.add_argument('--progress_bar_refresh', type=int, default=10,
                        help='refresh rate for progress bar (default: 10)')                            
    parser.add_argument('--logger_name', type=str, default="tb_logs",
                        help='name used for director for logging (default: tb_logs)')     
    parser.add_argument("--model_type", type=str, default="mlp",
                        help='type of model to pick, either "mlp" or "resnet" (default mlp)')                                           
    args = parser.parse_args()    
    return args


def fetch_lenta_tr_val(random_state=None):
    lenta_data = data.FetchLenta()
    
    X_train, X_val, trmnt_train, trmnt_val, y_train, y_val = lenta_data.train_test_split(random_state=random_state)

    m, s = X_train.mean(axis=0), X_train.std(axis=0)    
    
    tr_set = data.TreatmentDataset((X_train-m)/s, y_train.values, trmnt_train.values)
    val_set = data.TreatmentDataset((X_val-m)/s, y_val.values, trmnt_val.values)
    
    return tr_set, val_set
        

if __name__ == "__main__":  # pragma: no cover
    args = cli_main()
    logger = TensorBoardLogger(args.logger_name, name=args.name)
    logger.log_hyperparams(vars(args))
    
    train_set, val_set = fetch_lenta_tr_val(args.seed)    
    tr_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)    
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)    
    
    for batch in val_loader:
        break
    _, input_dim = batch[0].size()

    if args.model_type == 'mlp':
        net_arch = UpliftMLP
    elif args.model_type == 'resnet':
        net_arch = UpliftResNet
    else:
        raise Exception('Model type not supported')
    
    mlp = net_arch(input_dim=input_dim, output_dim=1, hidden_dim=args.nhid, num_hidden_layers=args.layers,
                           l2_weight=args.l2, l2_diff=args.l2_diff, learning_rate=args.lr, 
                           optimizer=args.optim, gard_clip=args.grad_clip, lr_scheduler=args.lr_scheduler,
                           use_layer_norm=args.layer_norm
                        )
    
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, max_steps=args.max_steps, 
                            accelerator=args.accelerator, 
                            callbacks=[TQDMProgressBar(refresh_rate=args.progress_bar_refresh)]
                        )
    trainer.fit(model=mlp, train_dataloaders=tr_loader, val_dataloaders=val_loader)
    
    
