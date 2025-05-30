import argparse
from dask.distributed import LocalCluster
from gev_nonstat_loc_utils import fit_ns_gev_single as fit_ns_gev_single_loc
from gev_nonstat_locscale_utils import fit_ns_gev_single as fit_ns_gev_single_locscale


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=str, required=True)
    parser.add_argument("--gcm", type=str, required=True)
    parser.add_argument("--member", type=str, required=True)
    parser.add_argument("--ssp", type=str, required=True)
    parser.add_argument("--metric_id", type=str, required=True)
    parser.add_argument("--bootstrap", type=int, required=False, default=0)
    parser.add_argument("--scale", type=int, required=False, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scale == 0:
        fit_ns_gev_single_loc(
            ensemble=args.ensemble,
            gcm=args.gcm,
            member=args.member,
            ssp=args.ssp,
            metric_id=args.metric_id,
            bootstrap=bool(args.bootstrap),
        )
    else:
        fit_ns_gev_single_locscale(
            ensemble=args.ensemble,
            gcm=args.gcm,
            member=args.member,
            ssp=args.ssp,
            metric_id=args.metric_id,
            bootstrap=bool(args.bootstrap),
        )


if __name__ == "__main__":
    # Load Dask cluster
    cluster = LocalCluster(n_workers=20, threads_per_worker=1)
    client = cluster.get_client()

    main()

    # Close Dask cluster
    cluster.close()
