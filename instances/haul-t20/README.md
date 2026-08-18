# Instance `haul-t20`

**SYNTHETIC -- not derived from any operating mine.** See `metadata.json` for the full provenance statement.

This instance belongs to the haulage-ratio axis, which holds the rock mass, the portal and the orebody geometry completely fixed and varies only the tonnage moved over the network -- and therefore only the ratio of construction cost to operating cost. Nothing in the search reads tonnage, so every member of this group has identical geometry, identical construction cost and identical buildability verdicts; only the `total` track differs. The layout is two plan-separated clusters at interleaved depths, the geometry in which a depth-ordered chain must cross between clusters at every level while a tree branches once. Within that group it varies `tonnage_scale` = `2.0`; every other parameter is held identical to the rest of the group, with `haul-t10` as the reference member.

The rock mass is competent ground (median Q around 8), where support cost is dominated by the cheapest tier and the faults and barrier supply nearly all the cost contrast, cut by no faulting at all, giving a clean baseline. A low-Q barrier slab spans RL 340-400 m and must be crossed by any decline; two competent windows breach it, posing a choice between a lateral detour to a breach and driving straight through the lid. The grid is 126 x 82 x 100 = 1,033,200 voxels at a 5 m cell size, of which 88.6% lie below the ground surface and are excavatable.

The network must connect one portal (P_South, 19 candidate voxels) to 4 production zone(s) distributed over 4 sublevel(s), which makes 12 of the 7 topology families applicable: `sublevel_fan`, `two_branch`, `three_branch`, `single_junction`, `sequential_ramp`, `chained_fan`, `hybrid_chained_fan_branch`, `spiral_decline`, `switchback_decline`, `twin_decline`, `conventional_decline`, `steiner_insertion`.

## Cost model

Edge weight is `w(u,v) = (cost_grid[v] + excavation_rate_per_m) * effective_length_m(u,v)` with `excavation_rate_per_m` = $1,000/m, which is **not** baked into `cost_grid.npz`. Support cost per metre follows the six-tier NGI Barton Q schedule; fault intercepts impose a floor of $3,059.0/m (single) or $4,460.9/m (multiple).

```python
from loader import load_instance
inst = load_instance('haul-t20')
inst['cost_grid']   # float32 $/m, 1e9 = not excavatable
inst['reference']   # reference_type, reference_cost, lower_bound
```
