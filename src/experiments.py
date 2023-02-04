from argparse import ArgumentParser
import data
from models.uplift_mlp import UpliftMLP

import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')    
    parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='num epochs (default: 1111)')  
    parser.add_argument('--l2', type=float, default=1e-3,
                    help='l2 regularization for each learner')    
    parser.add_argument('--l2_diff', type=float, default=1e-3,
                    help='l2 regularization for both learners')        
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')  
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')    
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units per layer (default: 256)')    
    parser.add_argument('--layers', type=int, default=1,
                        help='# of levels (default: 1)')
    parser.add_argument('--name', type=str, default="my_model",
                        help='name used for logging (default: my_model)')                        
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
    logger = TensorBoardLogger("tb_logs", name=args.name)
    logger.log_hyperparams(vars(args))
    
    train_set, val_set = fetch_lenta_tr_val(args.seed)    
    tr_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)    
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)    
    
    for batch in val_loader:
        break
    _, input_dim = batch[0].size()
    
    mlp = UpliftMLP(input_dim=input_dim, output_dim=1, hidden_dim=args.nhid, num_hidden_layers=args.layers,
                           l2_weight=args.l2, l2_diff=args.l2_diff, learning_rate=args.lr, optimizer = args.optim
                        )
    
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs)
    trainer.fit(model=mlp, train_dataloaders=tr_loader, val_dataloaders=val_loader)
    
    
