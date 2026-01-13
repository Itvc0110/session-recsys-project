import argparse
import yaml
import importlib
import torch
from data.datasets.datasets import SequentialDataset  # Assume generalized
from data.utils import create_dataloader
from src.trainers import Trainer
from src.evaluators import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--tune', action='store_true')  
    args = parser.parse_args()

    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(f'configs/{args.dataset}/{args.model}.yaml', 'r') as f:
        specific_config = yaml.safe_load(f)
    config.update(specific_config)  

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.epochs: config['epochs'] = args.epochs


    dataset_module = importlib.import_module(f"data.datasets.{args.dataset}")
    dataset_class = getattr(dataset_module, "SequentialDataset")
    dataset = dataset_class(config['data_path'], config.get('sep', ','), config['max_seq_len'], config['min_interactions'])

    train_seqs, valid_seqs, test_seqs = dataset.split()
    train_dl = create_dataloader(train_seqs, config['batch_size'], True, dataset.num_items, config['neg_samples'])
    valid_dl = create_dataloader(valid_seqs, config['batch_size'], False, dataset.num_items, config['neg_samples'])

    config['num_items'] = dataset.num_items

    model_names = {'sasrec': 'SASRec', 'gru4rec': 'GRU4Rec'}
    model_module = importlib.import_module(f"models.{args.model}")
    model_class = getattr(model_module, model_names[args.model])
    model = model_class(config)

    evaluator = Evaluator(config)
    trainer = Trainer(model, config)
    best_score, last_valid_result = trainer.fit(train_dl, valid_dl, evaluator)
    print(f"Best NDCG@10: {best_score:.4f}")
    print("All evaluation metrics on val set (last check):")
    for metric in config['eval_metrics']:
        for k in config['eval_k_values']:
            key = f"{metric}@{k}"
            print(f"{key}: {last_valid_result[key]:.4f}")

if __name__ == "__main__":
    main()