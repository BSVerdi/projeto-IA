import csv

import numpy as np


def read_results(file_path):
    results = {
        'test_loss': [],
        'test_acc': [],
        'macro_f1': [],
        'normal_f1': [],
        'pneumonia_f1': []
    }

    with open(file_path, mode='r') as f:
        with open(file_path, mode='r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                results['test_loss'].append(float(row['test_loss']))
                results['test_acc'].append(float(row['test_acc']))
                results['macro_f1'].append(float(row['macro_f1']))
                results['normal_f1'].append(float(row['normal_f1']))
                results['pneumonia_f1'].append(float(row['pneumonia_f1']))

        return results


def main():
    results = read_results('../results/cnn_results.csv')

    for c in results:
        print(f'Média do {c}: {np.mean(results[c]):.2f}%')


if __name__ == '__main__':
    main()
