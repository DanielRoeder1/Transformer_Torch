import enum
from transformers import AutoTokenizer, PretrainedConfig
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import wandb
import torch

from data import Wmt14Handler
from model import Transformer
from utils import get_time, AverageMeter

def train(model, train_loader, val_loader, optimizer, config):
    for i in range(config.train_epochs):
        print(f"[{get_time()}] Starting training Epoch {i}")
        train_epoch(model, train_loader, val_loader, optimizer, config)
        print(f"[{get_time()}] Finished training Epoch {i}")

    print(f"[{get_time()}] Starting final validation")
    validate(model, val_loader, config)

def validate(model, loader, config):
    av = AverageMeter()
    model.eval()

    for batch_idx , data in enumerate(loader):
        src_input, trgt_seq = data["input_ids"].cuda(), data["labels"].cuda()
        trgt_input = trgt_seq[:,:-1]
        trgt_label = trgt_seq[:,1:]

        with torch.no_grad():
            pred = model(src_input,trgt_input)
        
        loss, accuracy = calc_loss(pred, trgt_label,config.pad_idx)
        av.update(loss.item(), accuracy)

    avg_loss , avg_accuracy = av.get_avg()
    if avg_loss < best_val_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), config.save_path+f"/{avg_loss:.3f}.pth")

    model.train()
    print(f"[{get_time()}] Finished validation: Loss {avg_loss}, Accuracy {avg_accuracy}")


def train_epoch(model, train_loader, val_loader, optimizer, config):
    av = AverageMeter()
    model.train()

    for batch_idx, data in enumerate(train_loader):
        src_input, trgt_seq = data["input_ids"].cuda(), data["labels"].cuda()
        trgt_input = trgt_seq[:,:-1]
        trgt_label = trgt_seq[:,1:]

        optimizer.zero_grad()
        pred = model(src_input, trgt_input)
        # Permute to [batch, num_classes, seq_len]
        loss, accuracy = calc_loss(pred, trgt_label, config.pad_idx)
        av.update(loss.item(), accuracy)
        loss.backward()
        optimizer.step_and_update()

        if batch_idx % config.print_freq == 0:
            avg_loss, avg_accuracy = av.get_avg()
            print(f"[{batch_idx}/{len(train_loader)}] Loss: {loss.item():.3f} ({avg_loss:.3f}, Accuracy: {accuracy:.3f} ({avg_accuracy}))")
        wandb.log({"loss": loss.item(), "lr": opitmizer.optimizer.param_groups[0]["lr"], "accuracy_batch": accuracy, "ppl": torch.exp(loss).item()})
    
        if batch_idx % config.validate_every == 0 and batch_idx != 0:
            validate(model, val_loader, config)

def calc_loss(pred, target, pad_idx):
    greedy_prediction = pred.argmax(2)
    # Num_words_correct / Num_words not padding in batch
    accuracy = (greedy_prediction.eq(target).masked_select(target != config.pad_idx).sum() / (target != config.pad_idx).sum()).item()
    loss = cross_entropy(pred.permute([0,2,1]), target, ignore_index= pad_idx)
    return loss, accuracy


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

    
# TODO Implement validation
# TODO Implement Model saving


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("rossanez/t5-small-finetuned-de-en-lr2e-4")
    tokenizer.add_special_tokens({"bos_token": "<start>"})

    config = PretrainedConfig.from_json_file("config.json")
    # Tokenizer.vocab_size is not updating when adding new tokens 
    config.update({"vocab_size": len(tokenizer), "pad_idx": tokenizer.pad_token_id})

    wandb.init(project = "Transformer_training", config = config.to_dict())
    model = Transformer(config).cuda()
    wandb.watch(model)

    opitmizer = Scheduler(Adam(model.parameters(), betas=(0.9,0.98), eps= 10e-9), config)

    Dataset = Wmt14Handler(tokenizer, config, "de-en").get_wmt14()
    train_loader = DataLoader(Dataset["train"], batch_size= config.batch_size, shuffle= config.shuffle)
    val_loader = DataLoader(Dataset["val"], batch_size= config.batch_size, shuffle= config.shuffle)
    
    best_val_loss = 1e10
    train(model, train_loader,val_loader, opitmizer, config)



