"""TB-Trust: trustworthy tuberculosis screening under smartphone-capture degradation.

Sub-packages:
    data   - dataset manifest, leave-one-clinic-out splits, smartphone degradation pipeline
    models - baseline classifier, TB-Net reproduction slot, uncertainty / deferral heads
    train  - training loop
    eval   - calibration, safe-deferral (risk-coverage), cross-site analysis, metrics
    utils  - seeding, io helpers
"""

__version__ = "0.1.0"
