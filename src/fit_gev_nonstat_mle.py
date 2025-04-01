import argparse
from dask.distributed import LocalCluster
from gev_nonstat_utils import fit_ns_gev_single


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=str, required=True)
    parser.add_argument("--gcm", type=str, required=True)
    parser.add_argument("--member", type=str, required=True)
    parser.add_argument("--ssp", type=str, required=True)
    parser.add_argument("--metric_id", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    fit_ns_gev_single(
        ensemble=args.ensemble,
        gcm=args.gcm,
        member=args.member,
        ssp=args.ssp,
        metric_id=args.metric_id,
    )


if __name__ == "__main__":
    # Load Dask cluster
    cluster = LocalCluster()
    client = cluster.get_client()

    main()

    # Close Dask cluster
    cluster.close()
