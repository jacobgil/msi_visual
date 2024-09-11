import argparse
from msi_visual.extract.bruker_tsf_to_numpy import BrukerTsfToNumpy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_mz', type=int, default=300)
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--end_mz', type=int, default=1300)
    parser.add_argument('--input_path', type=str, required=True,
                        help='.d folder')
    parser.add_argument('--nonzero', action='store_true', default=False,
                        help='Save only m/zs that have a non zero value anywhere')

    parser.add_argument('--output_path', type=str, required=True,
                        help='Where to store the output .npy files')
    parser.add_argument(
        '--bins', type=int, default=1,
        help='How many bins per m/z value')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Parallel processes')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
    extraction = BrukerTsfToNumpy(args.start_mz, args.end_mz, args.bins, args.nonzero, args.id)
    extraction(args.input_path, args.output_path)
