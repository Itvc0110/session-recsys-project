import argparse
import yaml
import importlib
from data.datasets.ml1m import SequentialDataset
from data.utils import create_dataloader
from src.trainers import Trainer
from src.evaluators import Evaluator
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True) 
    parser.add_argument('--dataset', type=str, default='ml1m')
    args = parser.parse_args()

    with open(f'configs/{args.dataset}/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('configs/base.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    config.update(base_config)
    
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = SequentialDataset(config['data_path'], config['sep'], config['max_seq_len'], config['min_interactions'])
    train_seqs, valid_seqs, test_seqs = dataset.split()
    train_dl = create_dataloader(train_seqs, config['batch_size'], True, dataset.num_items, config['neg_samples'])
    valid_dl = create_dataloader(valid_seqs, config['batch_size'], False, dataset.num_items, config['neg_samples'])

    config['num_items'] = dataset.num_items
    model_module = importlib.import_module(f"models.{args.model}")
    model_names = {'sasrec': 'SASRec', 'gru4rec': 'GRU4Rec'}
    model_class = getattr(model_module, model_names[args.model])
    model = model_class(config)

    evaluator = Evaluator(config)
    trainer = Trainer(model, config)
    best_score = trainer.fit(train_dl, valid_dl, evaluator)
    print(f"Best {config['valid_metric']}: {best_score}")

if __name__ == "__main__":
    main()



