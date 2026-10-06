"""单细胞数据研究平台：质控筛选、归一化、版本化结果保存。"""

from .model import CountMatrix
from .qc import QCParams, QCReport, filter_cells
from .normalize import NormalizeParams, normalize
from .compare import CellChange, QCComparison, compare_qc
from .store import PublishRejectedError, ResultStore, validate_result_identities
from .io import load_matrix
from .joint import (
    JOIN_MODES,
    JoinModeComparison,
    JointMatrix,
    cell_key,
    compare_join_modes,
    join_batches,
)
from .association import (
    ASSOCIATION_MODES,
    AssociationDecision,
    CellCandidate,
    MeasurementAssociation,
    associate_measurements,
    association_candidates,
)
from .pairing import (
    PAIR_BASIS,
    PairedMatch,
    PairingCandidate,
    PairingConflictError,
    PairingPlan,
    approve_pairs,
    pairing_candidates,
    solve_pairing,
)

__all__ = [
    "CountMatrix",
    "QCParams",
    "QCReport",
    "filter_cells",
    "NormalizeParams",
    "normalize",
    "CellChange",
    "QCComparison",
    "compare_qc",
    "ResultStore",
    "PublishRejectedError",
    "validate_result_identities",
    "load_matrix",
    "JOIN_MODES",
    "JointMatrix",
    "JoinModeComparison",
    "cell_key",
    "join_batches",
    "compare_join_modes",
    "ASSOCIATION_MODES",
    "AssociationDecision",
    "CellCandidate",
    "MeasurementAssociation",
    "associate_measurements",
    "association_candidates",
    "PAIR_BASIS",
    "PairedMatch",
    "PairingCandidate",
    "PairingConflictError",
    "PairingPlan",
    "approve_pairs",
    "pairing_candidates",
    "solve_pairing",
]
