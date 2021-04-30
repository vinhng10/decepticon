# Python Import:
import yaml
import torch
from torch.nn import functional as F
import numpy as np
from argparse import ArgumentParser
from metrics.metrics import Input, Metrics
# Pytorch Lightning Import:
import pytorch_lightning as pl
from transformers import BertForSequenceClassification, AutoTokenizer

# Internal Import:
from data.data import RaceDataModule
from utils.utils import (
    t5_collate_fn, t5_dis_collate_fn,
    transformer_collate_fn,
    rnn_batch_fn, rnn_dis_batch_fn,
    display_result_as_string,
    serialize_config
)


if __name__ == "__main__":
    # Choose the model
    # from models.transformer import RaceModule
    # from models.rnn import RaceModule
    from models.t5 import RaceModule

    batch_fn = None
    collate_fn = t5_collate_fn
    NUM_SAMPLES = 5

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = RaceDataModule.add_model_specific_args(parser)
    parser = RaceModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    config = yaml.load(open("../configs/t5.yaml"), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    fx_dm = RaceDataModule(args, collate_fn)
    fx_dm.prepare_data()
    fx_dm.setup()
            # Trainer:
    trainer = pl.Trainer.from_argparse_args(args)
    fx_model = RaceModule.load_from_checkpoint("models/ckpts/t5.ckpt")
    fx_model.setup_tune(top_p = 0.95, top_k = 50, no_repeat_ngram_size = 2, num_samples = NUM_SAMPLES)
    
    
#     qj_model = BertForSequenceClassification.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator").cuda()
#     qj_tokenizer  = AutoTokenizer.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator")
    
    fx_model.eval()
#     qj_model.eval()
    
    metrics = Metrics()
    
    summary = {"bleu_1": 0.0,
            "bleu_2": 0.0,
            "bleu_3": 0.0,
            "bleu_4": 0.0,
            "meteor": 0.0,
            "rouge_l": 0.0}
    
    count = 0
    print("Total Length", len(fx_dm.test_dataloader()))
    
    for x, y in fx_dm.test_dataloader():
        output = fx_model.generate(x)

        y["input_ids"] = torch.repeat_interleave(y["input_ids"], NUM_SAMPLES, dim = 0)

        predictions = [
            fx_model.tokenizer.decode(generation, skip_special_tokens=True)
            for generation in output
        ]

        references = [
            fx_model.tokenizer.decode(target, skip_special_tokens=True)
            for target in y["input_ids"]
        ]
        
#         context = [ fx_model.tokenizer_.decode(c, skip_special_tokens=False\
#                                               ).replace("[ANS]", "").split("[CON]")[0] \
#                    for c in torch.repeat_interleave(x["input_ids"], NUM_SAMPLES, dim = 0)]
        
#         x2 = zip(predictions, context)
#         inputs = []
        
#         for item in x2:
#             inputs.append(" ".join(["[CLS]", item[0], "[SEP]", item[1]]))
#         inputs = qj_tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
#         output2 = F.softmax(qj_model.forward(inputs["input_ids"].cuda())["logits"], dim=1).squeeze()[:,0]
#         good_args = torch.nonzero(output2 > 0.99).cpu().numpy().squeeze()
        
#         predictions = [predictions[i] for i in good_args]
#         references = [references[i] for i in good_args]
        
        # Compute metrics:
        inputs = Input(predictions=predictions, references=references)
        m = metrics.compute_metrics(inputs)
        for key, value in m.items():
            summary[key] += value
        count+=1
        if count % 25 == 0: 
            print(count, {k: v / count for k, v in summary.items()})


    print("end", {k: v / count for k, v in summary.items()})
        
    
    
    
    
    
    
    #trainer.test(fx_model, test_dataloaders=fx_dm.test_dataloader())