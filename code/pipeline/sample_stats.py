"""Summarize saved Stim detector samples for each round count."""
import argparse
import csv
from pathlib import Path
import numpy as np

def calculate_statistics(file_path: Path, rounds: int) -> dict:
    with np.load(file_path) as data:
        dets = data['dets']
        obs = data['obs']
        if dets.ndim != 2 or obs.ndim != 2 or obs.shape != (len(dets), 1):
            raise ValueError(f'Unexpected data shape in {file_path}')
        n, d = dets.shape
        fired = dets.sum(axis=1)
        return {
            'rounds': rounds,
            'shots': n,
            'detectors': d,
            'avg_fired': float(fired.mean()),
            'std_fired': float(fired.std()),
            'empty_pct': float(100 * np.mean(fired == 0)),
            'detector_rate_pct': float(100 * fired.sum() / dets.size),
            'logical_flip_pct': float(100 * np.mean(obs[:, 0] == 1)),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--min-rounds', type=int, default=3)
    parser.add_argument('--max-rounds', type=int, default=100)
    args = parser.parse_args()
    results = []
    for r in range(args.min_rounds, args.max_rounds + 1):
        file_path = args.run_dir / args.split / f'r{r:03d}.npz'
        if not file_path.exists():
            print(f'Skipping r={r}: file not found')
            continue
        stats = calculate_statistics(file_path, r)
        results.append(stats)
        print(f"r={r:3d} | shots={stats['shots']:6d} | detectors={stats['detectors']:4d} | "
              f"avg fired={stats['avg_fired']:.2f} | empty={stats['empty_pct']:.2f}% | "
              f"logical flip={stats['logical_flip_pct']:.2f}%")
    if not results:
        print('No matching round data found.')
        return
    output = args.run_dir / f'sample_statistics_{args.split}.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'\nSaved statistics to: {output}')

if __name__ == '__main__':
    main()