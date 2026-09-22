"""Run from XPolicyLab's policy/Pi_05/openpi directory with YAM on PYTHONPATH."""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['stats', 'batch', 'train'])
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--exp-name', default='yam')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    from deployment.openpi_config import make_config
    cfg = make_config(args.repo_id, args.exp_name, args.batch_size)
    if args.mode == 'batch':
        from openpi.training import data_loader
        batch = next(iter(data_loader.create_data_loader(cfg, num_batches=1, shuffle=False)))
        import jax
        print(jax.tree.map(lambda value: (value.shape, str(value.dtype)), batch))
        return
    path = Path('scripts') / ('compute_norm_stats.py' if args.mode == 'stats' else 'train.py')
    if not path.exists():
        raise SystemExit('Run from the XPolicyLab policy/Pi_05/openpi directory')
    spec = importlib.util.spec_from_file_location('yam_openpi_entry', path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    if args.mode == 'stats':
        # Upstream statistics CLI resolves its config by name.
        from openpi.training import config
        original = config.get_config
        config.get_config = lambda name: cfg if name == cfg.name else original(name)
        entry.main(cfg.name)
    else:
        entry.main(cfg)

if __name__ == '__main__':
    main()
