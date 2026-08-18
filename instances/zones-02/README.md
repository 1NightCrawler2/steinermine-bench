# Instance `zones-02`

**SYNTHETIC -- not derived from any operating mine.** See `metadata.json` for the full provenance statement.

This instance belongs to the zone-count sweep, which holds the geology and portal fixed and varies only how many production terminals the network must reach. Within that group it varies `n_zones` = `2`; every other parameter is held identical to the rest of the group, with `xgrid-f2x-qpoor` as the reference member.

The rock mass is poor-dominated ground (median Q around 0.15), where most of the rock mass sits in the expensive tiers and the competent windows become decisive, cut by two faults that cross near the centre of the domain, producing a multi-intercept core where the higher $4,460.9/m floor applies. A low-Q barrier slab spans RL 340-400 m and must be crossed by any decline; two competent windows breach it, posing a choice between a lateral detour to a breach and driving straight through the lid. The grid is 126 x 82 x 100 = 1,033,200 voxels at a 5 m cell size, of which 88.6% lie below the ground surface and are excavatable.

The network must connect one portal (P_South, 19 candidate voxels) to 2 production zone(s) distributed over 2 sublevel(s), which makes 9 of the 7 topology families applicable: `sublevel_fan`, `two_branch`, `single_junction`, `sequential_ramp`, `chained_fan`, `spiral_decline`, `switchback_decline`, `conventional_decline`, `steiner_insertion`.

## Cost model

Edge weight is `w(u,v) = (cost_grid[v] + excavation_rate_per_m) * effective_length_m(u,v)` with `excavation_rate_per_m` = $1,000/m, which is **not** baked into `cost_grid.npz`. Support cost per metre follows the six-tier NGI Barton Q schedule; fault intercepts impose a floor of $3,059.0/m (single) or $4,460.9/m (multiple).

```python
from loader import load_instance
inst = load_instance('zones-02')
inst['cost_grid']   # float32 $/m, 1e9 = not excavatable
inst['reference']   # reference_type, reference_cost, lower_bound
```
