"""python -m fedup_repro.smoke   -> runs every scenario and method on synthetic data (CPU, ~1 minute). Run this first."""
import os
import tempfile


def main():
    # keep the smoke test away from real data / results
    tmp = tempfile.mkdtemp(prefix="fedup_smoke_")
    for k in ("FEDUP_DATA_DIR", "FEDUP_ARTIFACTS_DIR", "FEDUP_ROOT"):
        os.environ.setdefault(k, tmp)
    from . import experiments as E
    from .paths import ensure_dirs
    ensure_dirs()
    done, failed = E.run_jobs(E.plan("smoke", seeds=(0,)), device="cpu", force=True, verbose=False)
    for j in done:
        print("ok", j["scenario"], j["method"])
    assert not failed
    from . import aggregate as A
    df = A.runs_df("smoke")
    assert len(df) == len(done), (len(df), len(done))
    print(f"smoke test passed: {len(done)} runs, tables built from {len(df)} rows (temp dir {tmp})")


if __name__ == "__main__":
    main()
