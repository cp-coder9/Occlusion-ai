import argparse
import yaml
from app.pipeline import Pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    p = Pipeline(cfg)
    p.run()
