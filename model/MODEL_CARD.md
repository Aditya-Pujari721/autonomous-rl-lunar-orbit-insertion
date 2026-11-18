# Frozen nominal lunar capture benchmark v1


Policy file: policy.pt
SHA256: fbe1e712d21d915544af0baf099b3cbc22280932c07267a2faca5d58fdc47356
Source: runs/recorded_capture/s3/recorded_policy.pt
Inference uses a deterministic neural policy to select coast or thrust, action duration, and thrust direction.

## Independent confirmation

Named checkpoint fixed before testing. 512 fresh incoming states, disjoint from recorded prior training/evaluation seeds.
- Verified captures: 478/512 (93.36%).
- Lunar/physical impacts: 26/512.
- Other failures: 8/512.
- Wilson 95% interval for capture rate under this synthetic distribution: 90.86--95.21%.
- Halving integration steps with matched command replay changed zero capture/impact classifications across all 512 flights.
- Successful flights ending with empty propellant: 478/478.

The 113/128 development result informed checkpoint selection. The 512-flight confirmation measured performance after selection.

## Capture conditions

Starts are unbound incoming states 10--14 Moon radii away, hyperbolic excess speed 400--1000 m/s, random 3D orientation and lunar phase, tangential speed fraction 0.2--0.45, and 50--100% initial Apollo-inspired SPS propellant.

Capture requires osculating perilune altitude >=50 km and apolune altitude <=10,000 km for six hours, plus an independent 24-hour unpowered coast in the same bounds. Exact navigation and engine execution; finite thrust and changing mass. Circular lunar ephemeris, spherical gravity and idealized instantaneous attitude steering.
