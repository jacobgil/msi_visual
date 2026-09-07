import argparse
import logging
import sys

from msi_visual.extract.pymzml_to_numpy import PymzmlToNumpy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_mz', type=int, default=None)
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--end_mz', type=int, default=None)
    parser.add_argument('--input_path', type=str, required=True,
                        help='.imzML path (or legacy .d folder for other extractors)')
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
                        help='Optional row slice for fast backend (debug/benchmark)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Line-buffered logs so long extracts don't look stuck.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    args = get_args()
    logging.info("args=%s", args)
    if args.fast:
        if args.nonzero:
            raise SystemExit("fast backend does not support --nonzero yet; use --no-fast")
        from msi_visual.extract.imzml_fast import ImzmlFastExtractor
        extraction = ImzmlFastExtractor(
            id=args.id,
            min_mz=args.start_mz,
            max_mz=args.end_mz,
            bins_per_mz=args.bins,
            num_workers=args.num_workers,
            max_rows=args.max_rows,
        )
        extraction(args.input_path, args.output_path)
    else:
        extraction = PymzmlToNumpy(
            id=args.id,
            min_mz=args.start_mz,
            max_mz=args.end_mz,
            bins_per_mz=args.bins,
            nonzero=args.nonzero,
        )
        extraction(args.input_path, args.output_path)
    logging.info("Finished.")
