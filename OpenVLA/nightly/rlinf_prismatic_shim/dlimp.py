"""Import-time stub for optional RLDS data-pipeline dependency.

RLinf's online OpenVLA path uses the batch transform, not the offline RLDS
dataset builder.  The real dlimp package is therefore unnecessary here; this
stub only satisfies the type annotation while keeping the source untouched.
"""
class DLataset:
    pass
