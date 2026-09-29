Domain routing characterization on existing traces (B=12, W=8).

Definitions (aligned with rebuttal analysis code where possible):

Window
  Non-overlapping blocks of W consecutive decode layer-events per MoE layer
  (same event stream as DiffMoETraceSource logical_manifest iteration).
  Primary characterization uses W=8.

routing_skew_gini
  Gini coefficient of per-expert routed-token totals aggregated over all
  compared decode events in the workload (0=uniform, 1=one expert).

consecutive_window_spearman
  Mean Spearman rank correlation of routed-token count vectors between
  consecutive non-overlapping W-windows, averaged over layers and window pairs.

consecutive_topB_jaccard
  Mean Jaccard similarity of Top-B expert sets (by routed-token count within
  each window) between consecutive windows; B=12.

topB_routed_token_coverage
  Mean, over windows, of (routed tokens assigned to the window's Top-B experts)
  / (all routed tokens in the window). Equivalent to load_topB_coverage in
  analyze_correlation_mechanism.py.

Policy coverages (Cumulative Weighted-LFU CSV key `WeightedLFU` / THER /
Oracle) use the same schedule builders
and warmup rule as the primary evaluate_policies path (B=12, W=8, F=4).
