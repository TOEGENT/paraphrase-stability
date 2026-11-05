"""Values shared by the dataset workflow, the model client and the dataset checks."""

EQUIVALENCE_STATUSES = ("equivalent", "non_equivalent", "unclear", "insufficient_context")
RESOLVED_STATUSES = ("equivalent", "non_equivalent")
UNRESOLVED_STATUSES = ("unclear", "insufficient_context")
# A question or answer the judge did not clearly call different.
POSSIBLE_MATCH_STATUSES = ("equivalent", *UNRESOLVED_STATUSES)

SPLITS = ("train", "val", "test")

# Fields copied from answer rows onto every anchor row.
ANCHOR_METADATA_KEYS = ("source", "topic", "language", "slm_model")
