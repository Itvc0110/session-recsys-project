import argparse
import yaml
import importlib
from data.datasets.base_dataset import SequentialDataset
from data.utils import create_dataloader
from src.evaluators import Evaluator
from src.helpers import load_checkpoint
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='ml1m')
    parser.add_argument('--checkpoint', type=str, default='experiments/checkpoints/model.pth')
    args = parser.parse_args()

    with open(f'configs/{args.dataset}/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open(f'configs/{args.dataset}/base.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    config.update(base_config)

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_module = importlib.import_module(f"data.datasets.{args.dataset}")
    dataset_class = getattr(dataset_module, "SequentialDataset")
    dataset = dataset_class(
        config['data_path'],
        config.get('sep', '::'),
        config['max_seq_len'],
        config['min_interactions']
    )

    _, _, test_seqs = dataset.split()
    test_dl = create_dataloader(test_seqs, config['batch_size'], False, dataset.num_items, config['neg_samples'])

    config['num_items'] = dataset.num_items
    model_module = importlib.import_module(f"models.{args.model}")
    model_names = {'sasrec': 'SASRec', 'gru4rec': 'GRU4Rec'}
    model_class = getattr(model_module, model_names[args.model])
    model = model_class(config)
    load_checkpoint(model, args.checkpoint)

    evaluator = Evaluator(config)
    results = evaluator.evaluate(test_dl, model)
    print(results)

if __name__ == "__main__":
    main()