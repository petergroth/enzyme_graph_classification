# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from torch_geometric.datasets import TUDataset

def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    data = TUDataset(
        root=output_filepath,
        name='ENZYMES',
        use_node_attr=True,
        use_edge_attr=True)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    data_dir = str(project_dir) + '/data/'
    main(output_filepath=data_dir)

