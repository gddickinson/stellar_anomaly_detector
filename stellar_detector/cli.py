"""Command-line interface for the Stellar Anomaly Detector v6.0."""

from __future__ import annotations

import argparse
import sys

from .core.models import CatalogSource, DetectionConfig
from .utils.logging_config import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stellar Anomaly Detector v6.0 — Advanced Technosignature Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--source",
        choices=["gaia", "hipparcos", "tycho2", "2mass", "allwise", "synthetic"],
        default="synthetic",
        help="Single catalog to analyze (default: synthetic)",
    )
    src.add_argument(
        "--multi",
        type=str,
        help="Comma-separated catalog list for multi-catalog analysis",
    )
    src.add_argument("--file", type=str, help="Path to a CSV catalog file")
    src.add_argument("--gui", action="store_true", help="Launch the GUI")

    # Analysis parameters
    parser.add_argument("--stars", type=int, default=2000, help="Stars per catalog (default: 2000)")
    parser.add_argument("--output", type=str, default="./stellar_analysis", help="Output directory")
    parser.add_argument("--ra", type=float, default=180.0, help="Center RA (degrees)")
    parser.add_argument("--dec", type=float, default=0.0, help="Center Dec (degrees)")
    parser.add_argument("--radius", type=float, default=5.0, help="Search radius (degrees)")

    # Thresholds
    parser.add_argument(
        "--outlier-threshold", type=float, default=None,
        help="Robust z-score (MAD) threshold used by the outlier detectors "
             "(default: DetectionConfig, 4.0)",
    )
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument(
        "--match-radius", type=float, default=None,
        help="--multi: cross-match radius in arcsec (default: DetectionConfig, 5.0)",
    )

    # Cache / output options
    parser.add_argument("--no-cache", action="store_true", help="Skip cache, force fresh download")
    parser.add_argument("--format", choices=["csv", "json", "votable"], default="csv")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser


def config_from_args(args: argparse.Namespace) -> DetectionConfig:
    """DetectionConfig from parsed CLI arguments.

    The analyzers read ``mad_threshold``; ``--outlier-threshold`` sets it together with
    ``outlier_threshold``, as the GUI presets do.
    """
    config = DetectionConfig(isolation_contamination=args.contamination)
    if args.outlier_threshold is not None:
        config.outlier_threshold = args.outlier_threshold
        config.mad_threshold = args.outlier_threshold
    return config


def load_multi(fetcher, args: argparse.Namespace, config: DetectionConfig, use_cache: bool = True):
    """--multi: fetch each catalogue and cross-match the others onto the first by position.

    One row per star of the first catalogue (e.g. Gaia), with the other catalogues' photometry
    (e.g. AllWISE W1-W4) added when a source lies within the match radius, so multi-band tests
    such as the Dyson-sphere SED fit see all bands on the same star.
    """
    from .core.constants import CATALOG_METADATA
    from .data.cross_match import merge_catalogs

    sources = [SOURCE_MAP[c.strip()] for c in args.multi.split(",")]
    frames = {
        s.value: fetcher.fetch(s, n_stars=args.stars, ra_center=args.ra, dec_center=args.dec,
                               radius_deg=args.radius, use_cache=use_cache)
        for s in sources
    }
    names = list(frames)
    radius = args.match_radius if args.match_radius is not None else config.cross_match_radius_arcsec
    epochs = {k: v["epoch"] for k, v in CATALOG_METADATA.items() if "epoch" in v}
    others = {n: frames[n] for n in names[1:]}
    return merge_catalogs(frames[names[0]], others, radius_arcsec=radius, epochs=epochs,
                          primary_name=names[0])


SOURCE_MAP = {
    "gaia": CatalogSource.GAIA_DR3,
    "hipparcos": CatalogSource.HIPPARCOS,
    "tycho2": CatalogSource.TYCHO2,
    "2mass": CatalogSource.TWOMASS,
    "allwise": CatalogSource.ALLWISE,
    "synthetic": CatalogSource.SYNTHETIC,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    logger = setup_logging(level=args.log_level, log_file="stellar_detector.log")

    if args.gui:
        _launch_gui()
        return

    config = config_from_args(args)

    from .data.fetcher import DataFetcher
    from .data.preprocessing import preprocess_catalog
    from .analysis.hr_diagram import HRDiagramAnalyzer
    from .analysis.stellar_lifetime import StellarLifetimeAnalyzer
    from .analysis.kinematics import KinematicsAnalyzer
    from .analysis.spectral import SpectralAnalyzer
    from .analysis.technosignature import TechnosignatureAnalyzer
    from .analysis.ml_pipeline import MLPipeline
    from .analysis.ensemble import EnsembleScorer
    from .utils.io import save_results, load_catalog_csv, export_report

    use_cache = not args.no_cache
    fetcher = DataFetcher(output_dir=args.output)

    # Load data
    if args.file:
        df = load_catalog_csv(args.file)
    elif args.multi:
        df = load_multi(fetcher, args, config, use_cache=use_cache)
    else:
        source = SOURCE_MAP[args.source]
        df = fetcher.fetch(source, n_stars=args.stars,
                           ra_center=args.ra, dec_center=args.dec,
                           radius_deg=args.radius,
                           use_cache=use_cache)

    logger.info("Loaded %d stars", len(df))

    # Preprocess
    df = preprocess_catalog(df, config)
    logger.info("After preprocessing: %d stars", len(df))

    # Run analysis modules
    all_results = []

    analyzers = [
        ("HR Diagram", HRDiagramAnalyzer(config)),
        ("Stellar Lifetime", StellarLifetimeAnalyzer(config)),
        ("Kinematics", KinematicsAnalyzer(config)),
        ("Spectral", SpectralAnalyzer(config)),
        ("Technosignature", TechnosignatureAnalyzer(config)),
        ("ML Pipeline", MLPipeline(config)),
    ]

    for name, analyzer in analyzers:
        logger.info("Running %s analysis...", name)
        try:
            results = analyzer.analyze(df)
            all_results.extend(results)
            logger.info("  -> %d anomalies", len(results))
        except Exception as e:
            logger.error("  -> %s failed: %s", name, e)

    # Ensemble aggregation
    scorer = EnsembleScorer()
    merged = scorer.aggregate(all_results)
    stats = scorer.summary_stats(merged)

    # Output
    save_results(merged, args.output, fmt=args.format)

    if args.report:
        export_report(merged, df, args.output)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Stellar Anomaly Detector v6.0 — Results Summary")
    print(f"{'='*60}")
    print(f"  Stars analyzed:     {len(df)}")
    print(f"  Total anomalies:    {stats['total']}")
    print(f"  Unique anomalous:   {stats.get('unique_stars', 0)}")
    print(f"  High priority:      {stats.get('high_priority_count', 0)}")
    print(f"  Mean confidence:    {stats.get('mean_confidence', 0):.2f}")
    print()

    if stats.get("by_type"):
        print("  Anomalies by type:")
        for t, c in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
            print(f"    {t:30s} {c}")

    if merged:
        print(f"\n  Top 5 anomalies:")
        for r in merged[:5]:
            print(f"    [{r.follow_up_priority}] {r.star_id}: {r.description[:60]}")

    print(f"\n  Results saved to: {args.output}/")
    print(f"{'='*60}\n")


def _launch_gui():
    """Launch the PySide6 GUI application."""
    try:
        from .gui.main_window import launch_app
        launch_app()
    except ImportError:
        print("GUI requires PySide6. Install with: pip install PySide6")
        print("Falling back to CLI mode. Use --help for CLI options.")
        sys.exit(1)


if __name__ == "__main__":
    main()
