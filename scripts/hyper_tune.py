import itertools
import yaml
import subprocess

params = {
    'learning_rate': [0.001, 0.0001],
    'dropout_prob': [0.1, 0.2, 0.3]
}

best_score = 0
best_params = {}
for combo in itertools.product(*params.values()):
    config = {'learning_rate': combo[0], 'dropout_prob': combo[1]}  
    with open('temp.yaml', 'w') as f:
        yaml.dump(config, f)
    result = subprocess.run(['python', 'scripts/train.py', '--config', 'temp.yaml', '--model', 'sasrec', '--dataset', 'ml1m'], capture_output=True)
    score = float(result.stdout.decode().split('Best validation score: ')[1])  
    if score > best_score:
        best_score = score
        best_params = config
print(best_params)
