
import numpy as np


stats = np.load('train_stats_frome.npy', allow_pickle=True).item()
changed = False

if 'means' in stats:
    stats['mean'] = stats.pop('means')
    changed = True

if 'stds' in stats:
    stats['std'] = stats.pop('stds')
    changed = True

if changed:
    np.save('train_stats_frome.npy', stats)
    print('train_stats_frome.npy key fixed to mean/std')
else:
    print('train_stats_frome.npy is already mean/std, no need to fix') 