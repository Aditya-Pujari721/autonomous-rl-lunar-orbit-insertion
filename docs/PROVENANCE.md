# Training history

The capture policy was trained in three PPO stages.

1. **Initial training** (`capture2_compressed`). Starting from random weights with seed 42, the policy trained for 256 updates. The selected checkpoint was `latest.pt`.
2. **Nominal capture** (`capture_nominal1`). Training continued for 64 updates with exact engine execution and physical-time advantages. A separate coast-duration output was initialized from the shared duration output. Development evaluation selected `selected.pt`.
3. **Final continuation** (`capture_fuel_insight1/control`). A further 32 updates used the capture reward with zero fuel penalty. The final `latest.pt` was selected before confirmation on 512 held-out arrivals.

The [stage records](provenance/) contain configurations, seed lists, and checkpoint selections. The viewer uses trajectories from a recorded repeat of training. Its final checkpoint is the [released policy](../model/policy.pt). It matched the original policy on all 512 confirmation cases.

| Checkpoint | SHA-256 |
|---|---|
| Original benchmark | `64881398fa46209899be093c68a1eb3714a44d243d8dcc5e833872b7d2ef4eb7` |
| Released policy | `fbe1e712d21d915544af0baf099b3cbc22280932c07267a2faca5d58fdc47356` |

The [model card](../model/MODEL_CARD.md) describes the confirmation results. The [README](../README.md) gives commands for repeating the evaluation and continuing training from the released policy.
