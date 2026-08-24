"""
research/pipeline.py
======================
Master research pipeline — executes Phase 12 through Phase 17 in order.

Usage:
    python -m research.pipeline                    # full run
    python -m research.pipeline --phases 12        # single phase
    python -m research.pipeline --phases 12 13 14  # selected phases
    python -m research.pipeline --fast             # skip slow tuning
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logger import get_logger

logger = get_logger(__name__)


def run_phase12():
    logger.info("─" * 50)
    logger.info("PHASE 12: BASELINE EVALUATION")
    logger.info("─" * 50)
    from research.phase12.prophet_eval import ProphetEvaluator
    from research.phase12.dqn_eval import DQNEvaluator
    from research.phase12.hars_eval import HARSEvaluator
    from research.phase12.cloudsim_eval import CloudSimEvaluator

    results = {}
    results["12A_prophet"]   = ProphetEvaluator().run()
    results["12B_dqn"]       = DQNEvaluator(n_eval_episodes=3).run()
    results["12C_hars"]      = HARSEvaluator().run()
    results["12D_cloudsim"]  = CloudSimEvaluator().run()
    logger.info("Phase 12 complete.")
    return results


def run_phase13(fast: bool = False):
    logger.info("─" * 50)
    logger.info("PHASE 13: HYPERPARAMETER TUNING")
    logger.info("─" * 50)
    if fast:
        logger.info("Fast mode: skipping Prophet and HARS tuning (DQN only).")

    from research.phase13.dqn_tuning import DQNTuner
    DQNTuner().run()

    if not fast:
        from research.phase13.prophet_tuning import ProphetTuner
        from research.phase13.hars_tuning import HARSTuner
        ProphetTuner().run()
        HARSTuner().run()

    logger.info("Phase 13 complete.")


def run_phase14():
    logger.info("─" * 50)
    logger.info("PHASE 14: ABLATION STUDY")
    logger.info("─" * 50)
    from research.phase14.ablation import AblationStudy
    AblationStudy().run()
    logger.info("Phase 14 complete.")


def run_phase15():
    logger.info("─" * 50)
    logger.info("PHASE 15: STATISTICAL VALIDATION")
    logger.info("─" * 50)
    from research.phase15.statistical_validation import StatisticalValidator
    StatisticalValidator().run()
    logger.info("Phase 15 complete.")


def run_phase16():
    logger.info("─" * 50)
    logger.info("PHASE 16: PUBLICATION VISUALIZATION")
    logger.info("─" * 50)
    from research.phase16.visualization import PublicationVisualizer
    PublicationVisualizer().run()
    logger.info("Phase 16 complete.")


def run_phase17():
    logger.info("─" * 50)
    logger.info("PHASE 17: DOCUMENTATION")
    logger.info("─" * 50)
    from research.phase17.documentation import DocumentationWriter
    DocumentationWriter().run()
    logger.info("Phase 17 complete.")


PHASE_MAP = {
    12: run_phase12,
    13: run_phase13,
    14: run_phase14,
    15: run_phase15,
    16: run_phase16,
    17: run_phase17,
}


def main():
    parser = argparse.ArgumentParser(
        description="Research Pipeline — Phase 12 to 17"
    )
    parser.add_argument(
        "--phases", nargs="+", type=int, default=list(range(12, 18)),
        help="Phases to run (default: 12 13 14 15 16 17)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip slow Prophet/HARS tuning (Phase 13)"
    )
    args = parser.parse_args()

    t_total = time.time()
    logger.info("=" * 60)
    logger.info("RESEARCH PIPELINE STARTED — Phases: %s", args.phases)
    logger.info("=" * 60)

    for phase in sorted(args.phases):
        if phase not in PHASE_MAP:
            logger.warning("Unknown phase %d — skipping.", phase)
            continue
        t0 = time.time()
        try:
            if phase == 13:
                PHASE_MAP[phase](fast=args.fast)
            else:
                PHASE_MAP[phase]()
            logger.info("Phase %d finished in %.1fs.", phase, time.time() - t0)
        except Exception as exc:
            logger.error("Phase %d FAILED: %s", phase, exc, exc_info=True)

    logger.info("=" * 60)
    logger.info("RESEARCH PIPELINE COMPLETE in %.1fs", time.time() - t_total)
    logger.info("Results: results/")
    logger.info("Reports: results/reports/")
    logger.info("=" * 60)

    _print_manifest()


def _print_manifest():
    """Print a final manifest of all generated outputs."""
    from config.settings import RESULTS_DIR
    logger.info("\n── OUTPUT MANIFEST ──")
    for path in sorted(Path(RESULTS_DIR).rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            suffix = path.suffix
            if suffix in (".md",".csv",".png",".pdf",".svg",".pt",".pkl"):
                logger.info("  %-12s  %7.1f KB  %s",
                            suffix, size/1024, path.relative_to(RESULTS_DIR))


if __name__ == "__main__":
    main()
