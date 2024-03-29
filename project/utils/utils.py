import torch
import numpy as np
from typing import List, Dict


rnn_test_batch_fn = lambda batch: (torch.cat([batch['answers']['input_ids'],batch['articles']['input_ids']], dim=1)[:2,:], torch.cat([batch['answers']['input_ids'],batch['articles']['input_ids']], dim=1)[:2,:], batch['questions']['input_ids'])
transformer_test_batch_fn = lambda batch: (batch['inputs']['input_ids'], {k:v[:2,:] for k, v in batch['inputs'].items()}, batch['targets']['input_ids'])


def default_collate_fn(batch, tokenizer):
    """"""
    articles = []
    questions = []
    answers = []
    distractors = []

    for item in batch:
        articles.append(item["article"])
        questions.append(item["question"])
        answers.append(item["answer"])
        distractors.append(tokenizer.additional_special_tokens[-1].join(item["distractors"]))

    return {
        "articles": tokenizer(articles, padding=True, truncation=True, max_length=500, return_tensors="pt"),
        "questions": tokenizer(questions, padding=True, return_tensors="pt"),
        "answers": tokenizer(answers, padding=True, return_tensors="pt"),
        "distractors": tokenizer(distractors, padding=True, return_tensors="pt"),
    }


def t5_collate_fn(batch, tokenizer):
    """"""
    context = []
    questions = []
    for item in batch:
        context.append(" ".join(["[ANS]", item["answer"], "[CON]", item["article"]]))
        questions.append(item["question"])
    context = tokenizer(text=context,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        pad_to_max_length=True,
                        max_length=512)
    questions = tokenizer(questions,
                          padding=True,
                          truncation=True,
                          return_tensors="pt",
                          pad_to_max_length=True,
                          max_length=512)

    context['input_ids'] = torch.squeeze(context['input_ids'])
    context['attention_mask'] = torch.squeeze(context['attention_mask'])
    questions['input_ids'] = torch.squeeze(questions['input_ids'])
    questions['attention_mask'] = torch.squeeze(questions['attention_mask'])

    return context, questions


def t5_dis_collate_fn(batch, tokenizer):
    """"""
    context = []
    distractor = []
    for item in batch:
        context.append(
            " ".join(["[ANS]", item["answer"], "[QUE]", item["question"], "[CON]", item["article"]]))
        indx = np.random.randint(low=0, high=len(item["distractors"]), size=1)[0]
        distractor.append(item["distractors"][indx])

    context = tokenizer(text=context,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        pad_to_max_length=True,
                        max_length=512)

    distractor = tokenizer(distractor,
                           padding=True,
                           truncation=True,
                           return_tensors="pt",
                           pad_to_max_length=True,
                           max_length=512)

    context['input_ids'] = torch.squeeze(context['input_ids'])
    context['attention_mask'] = torch.squeeze(context['attention_mask'])
    distractor['input_ids'] = torch.squeeze(distractor['input_ids'])
    distractor['attention_mask'] = torch.squeeze(distractor['attention_mask'])

    return context, distractor


def transformer_collate_fn(batch, tokenizer):
    """"""
    con_token, que_token, ans_token, dis_token = tokenizer.additional_special_tokens

    inputs = []
    targets = []

    for item in batch:
        inputs.append(" ".join([con_token, item["article"], ans_token, item["answer"]]))
        targets.append(" ".join([que_token, item["question"], dis_token, dis_token.join(item["distractors"])]))

    return {
        "inputs": tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"),
        "targets": tokenizer(targets, padding=True, truncation=True, return_tensors="pt"),
    }


def transformer_dis_collate_fn(batch, tokenizer):
    """"""
    con_token, que_token, ans_token, dis_token = tokenizer.additional_special_tokens

    inputs = []
    targets = []

    for item in batch:
        inputs.append(" ".join([item["answer"], tokenizer.sep_token, item['question'], tokenizer.sep_token, item["article"]]))
        targets.append(" ".join([tokenizer.sep_token.join(item["distractors"])]))

    return {
        "inputs": tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"),
        "targets": tokenizer(targets, padding=True, truncation=True, return_tensors="pt"),
    }


def rnn_collate_fn(batch, tokenizer):
    """"""
    con_token, que_token, ans_token, dis_token = tokenizer.additional_special_tokens

    inputs = []
    targets = []

    for item in batch:
        inputs.append(" ".join([item["answer"], tokenizer.sep_token, item["article"]]))
        targets.append(item['question'])

    return {
        "inputs": tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"),
        "targets": tokenizer(targets, padding=True, truncation=True, return_tensors="pt"),
    }


def rnn_dis_collate_fn(batch, tokenizer):
    """"""
    con_token, que_token, ans_token, dis_token = tokenizer.additional_special_tokens

    inputs = []
    targets = []

    for item in batch:
        inputs.append(" ".join([item["answer"], tokenizer.sep_token, item['question'], tokenizer.sep_token, item["article"]]))
        targets.append(" ".join([tokenizer.sep_token.join(item["distractors"])]))

    return {
        "inputs": tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"),
        "targets": tokenizer(targets, padding=True, truncation=True, return_tensors="pt"),
    }


def display_result_as_string(tokenizer, dataloader, model, test_batch_fc, pred_len=32, sample_num=1, f=None):
    """
    Args:
        test_batch_fc: function making batch into variables con and tgt,
        where con is the input of generate and tgt is the target sequence
        pred_len: int, length of generated sequences
        f: File write to some files
    """
    with torch.no_grad():
        for batch in dataloader:
            con, x, tgt = test_batch_fc(batch)
            out = model.generate(x, pred_len=pred_len)
            con = con[0, :].long().numpy()
            tgt = tgt[0, :].long().numpy()
            con_str = ' '.join(tokenizer.convert_ids_to_tokens(con, True))
            tgt_str = ' '.join(tokenizer.convert_ids_to_tokens(tgt, True))
            if f:
              f.write("CON:"+con_str+'\n')
              f.write("TGT:"+tgt_str+'\n')
              f.write("OUT:"+out_str+'\n')

            print("\n============================")
            print("CON:", con_str)
            print("TGT:", tgt_str)
            for i in range(sample_num):
                sample = out[i][0, :].long().numpy()
                out_str = ' '.join(tokenizer.convert_ids_to_tokens(sample, True))
                if f:
                    f.write("OUT%d:"%i + out_str + '\n')
                print("OUT%d:"%i, out_str)
            if f:
                f.write('=======================\n')


def serialize_config(config: Dict) -> List[str]:
    """"""

    # Get an empty list for serialized config:
    serialized_config = []

    for key, value in config.items():
        # Append key:
        serialized_config.append("--" + key)

        # Append value:
        if isinstance(value, int) or isinstance(value, float) or \
           isinstance(value, str) or isinstance(value, bool):
            serialized_config.append(str(value))
        elif isinstance(value, List):
            serialized_config += [str(val) for val in value]
        elif isinstance(value, bool):
            continue
        else:
            raise ValueError(f"Invalid value in config file: {value}")

    return serialized_config
