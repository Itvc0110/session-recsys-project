import itertools
import yaml
import subprocess

def hyper_tune(config, model, dataset):
    params = {
        'learning_rate': [0.001, 0.005, 0.0001],
        'dropout_prob': [0.2, 0.1, 0],
    }
    best_score = -float('inf')
    best_params = {}
    for lr, dropout in itertools.product(params['learning_rate'], params['dropout_prob']):
        print(f"Tuning with lr={lr}, dropout={dropout}")
        result = subprocess.run(['python', 'scripts/train.py', '--model', model, '--dataset', dataset, '--batch_size', str(bs), '--epochs', str(config['epochs'])], capture_output=True)
        output = result.stdout.decode()
        if 'Best NDCG@10' in output:
            score = float(output.split('Best NDCG@10: ')[1].strip())
            if score > best_score:
                best_score = score
                best_params = {'lr': lr, 'dropout': dropout}
    print(f"Best params: {best_params}, Best score: {best_score}")
    return best_params
