from transformers import AutoTokenizer, PretrainedConfig
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy

from data import Wmt14Handler
from model import Transformer
from utils import get_time, AverageMeter

def train_epoch(model, train_loader, optimizer, config):
    print(f"[{get_time()}] Starting Epoch")

    av = AverageMeter()
    model.train()

    for batch_idx, data in enumerate(train_loader):
        src_input, trgt_seq = data["input_ids"].cuda(), data["labels"].cuda()
        trgt_input = trgt_seq[:,:-1]
        trgt_label = trgt_seq[:,1:]

        optimizer.zero_grad()
        pred = model(src_input, trgt_input)
        # Permute to [batch, num_classes, seq_len]
        loss = calc_loss(pred.permute([0,2,1]), trgt_label, config.pad_idx)
        av.update(loss.item())
        loss.backward()
        optimizer.step_and_update()

        if batch_idx % config.print_freq == 0:
            print(f"[{batch_idx}/{len(train_loader)}] Loss: {loss.item():.3f} ({av.get_avg():.3f})")
    
    print(f"[{get_time()}] Finished Epoch")


def calc_loss(pred, target, pad_idx):
    return cross_entropy(pred, target, ignore_index= pad_idx)


class Scheduler():
    """
    Learning rate scheduler as described in publication
    Code inspired by: https://github.com/jadore801120/attention-is-all-you-need-pytorch.git
    """
    def __init__(self, optimizer, config) -> None:
        self.optimizer = optimizer
        self.hidden_size = config.hidden_size
        self.warmup_steps = config.warmup_steps

        self.steps = 0 
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step_and_update(self):
        self.update_learning_rate()
        self.optimizer.step()

    def update_learning_rate(self):
        self.n_steps = 1
        lr = (self.hidden_size **-0.5) * min((self.n_steps**-0.5), (self.n_steps * self.warmup_steps** (-1.5)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    
    


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("rossanez/t5-small-finetuned-de-en-lr2e-4")
    tokenizer.add_special_tokens({"bos_token": "<start>"})

    config = PretrainedConfig.from_json_file("config.json")
    # Tokenizer.vocab_size is not updating when adding new tokens 
    config.update({"vocab_size": len(tokenizer), "pad_idx": tokenizer.pad_token_id})

    model = Transformer(config).cuda()

    opitmizer = Scheduler(Adam(model.parameters(), betas=(0.9,0.98), eps= 10e-9), config)

    Dataset = Wmt14Handler(tokenizer, config, "de-en").get_wmt14()
    train_loader = DataLoader(Dataset, batch_size= config.batch_size, shuffle= config.shuffle)

    train_epoch(model, train_loader, opitmizer, config)



