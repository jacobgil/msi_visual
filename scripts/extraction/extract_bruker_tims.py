import argparse
import json
from msi_visual.extract.bruker_tims_to_numpy import BrukerTimsToNumpy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_mz', type=int, default=None)
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--end_mz', type=int, default=None)
    parser.add_argument('--mz_list', type=str, default=None)
    parser.add_argument('--input_path', type=str, required=True,
                        help='.d folder')
    parser.add_argument('--nonzero', action='store_true', default=False,
                        help='Save only m/zs that have a non zero value anywhere')

    parser.add_argument('--output_path', type=str, required=True,
                        help='Where to store the output .npy files')
    parser.add_argument(
        '--bins', type=int, default=1,
        help='How many bins per m/z value')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Parallel processes (fast backend only; default 8)')
    parser.add_argument(
        '--fast',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Streaming + np.add.at (+ workers) backend (default: on). Use --no-fast for legacy.',
    )
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Optional row slice for fast backend')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.mz_list:
        mz_list = list(map(float, json.load(open(args.mz_list, "r"))))
    else:
        mz_list = None

    if args.fast:
        if args.nonzero or mz_list is not None:
            raise SystemExit("fast backend does not support --nonzero / --mz_list yet; use --no-fast")
        if args.start_mz is None or args.end_mz is None:
            raise SystemExit("fast backend requires --start_mz and --end_mz")
        from msi_visual.extract.bruker_fast import BrukerFastExtractor
        extraction = BrukerFastExtractor(
            id=args.id,
            min_mz=args.start_mz,
            max_mz=args.end_mz,
            bins_per_mz=args.bins,
            num_workers=args.num_workers,
            max_rows=args.max_rows,
            kind="tims",
        )
        extraction(args.input_path, args.output_path)
    else:
        extraction = BrukerTimsToNumpy(
            id=args.id,
            min_mz=args.start_mz,
            max_mz=args.end_mz,
            mz_list=mz_list,
            bins_per_mz=args.bins,
            nonzero=args.nonzero,
        )
        extraction(args.input_path, args.output_path)
