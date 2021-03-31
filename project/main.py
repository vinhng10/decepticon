# Python Import:
from argparse import ArgumentParser

# Pytorch Lightning Import:
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from data.data import RaceDataModule
from models.rnn import RaceRNNModule


if __name__ == "__main__":
    pl.seed_everything(1234)

    parser = ArgumentParser()

    # --------------
    #  data args
    # --------------
    parser.add_argument("-f", "--file", required=False)
    parser.add_argument("--vocab_size", type=int, default=30522,
                        help="vocab size")
    parser.add_argument("--max_len_q", type=int, default=500,
                        help="Max length of question"),
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading. Set to 0 if running on Windows")
    parser.add_argument("--data_path", type=str, default='data/RACE_processed',
                        help="Path to data.")
    parser.add_argument("--is_preprocessed", type=bool, default=True,
                        help="Set it to False if the data requires preprocessing"),
    parser.add_argument("--batch_size", type=int, default=64)

    # --------------
    #  model args
    # --------------
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--pretrained_model", type=str, default="prajjwal1/bert-tiny")

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.gpus = '0'

    # ------------
    # data & model
    # ------------
    fx_dm = RaceDataModule(args)
    args.idx2w = {v: k for k, v in list(fx_dm.get_vocab().items())}
    fx_model = RaceRNNModule(args)

    # Callbacks:
    checkpoint = ModelCheckpoint(
        dirpath='models/ckpts/',
        filename="./fx-{epoch:02d}-{val_loss:.7f}",
        monitor="val_loss"
    )

    # Logger:
    logger = TensorBoardLogger('models/logs/')

    # Trainer:
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=1,
        checkpoint_callback=checkpoint,
        logger=logger
    )
    trainer.fit(fx_model, fx_dm)

    # fx_infer = RaceRNNModule.load_from_checkpoint(checkpoint.best_model_path)
    # fx_infer.eval()
    # print(fx_infer)

