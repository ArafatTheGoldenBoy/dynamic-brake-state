# Automotive Software Assessment

## Current strengths
- **Signal validation hooks**: Perception, planning, and actuation payloads define validation routines with units/freshness metadata, improving contract clarity for ECU-style messages. Fault codes for stale/invalid data are latched via the safety manager. 【F:ecu.py†L24-L170】【F:ecu.py†L381-L416】
- **Configurable in-process bus**: MessageBus simulates drop rate, jitter, and expiry metrics, providing a foundation for modeling non-ideal inter-ECU communication. 【F:ecu.py†L298-L371】
- **Calibration separation**: AEB calibration parameters load from JSON with checksum tagging, schema version metadata, and validation/clamping to keep tunables outside the control loop. 【F:calibrations.py†L1-L63】

## Research gaps and risks
1. **Signal contract completeness**
   - Units and ranges are declared but not enforced with plausibility relationships (e.g., TTC consistency, correlated confidence/visibility). Validity is largely boolean without freshness counters or diagnostic categories. 【F:ecu.py†L24-L170】
2. **Bus realism and supervision**
   - The bus models drop and jitter but lacks priority arbitration, deterministic scheduling, and watchdogs/deadline supervision; message loss is not fed into safety decisions beyond expiry counts. 【F:ecu.py†L298-L371】
3. **Safety management depth**
   - SafetyManager latches faults to force brake/throttle overrides but does not support degraded modes (e.g., torque limit, progressive ramps), reset policies, or fault provenance for auditability. 【F:ecu.py†L381-L416】
4. **Calibration integrity and lifecycle**
   - Calibration loader adds checksum and schema tag but lacks schema enforcement/validation against expected keys, signature verification, or environment selection (dev/test/prod). 【F:calibrations.py†L41-L63】
5. **Observability and traceability**
   - Metrics exist for the bus, but there is no structured logging of message contents, safety transitions, or control decisions with timestamps for post-incident analysis.
6. **Verification and test coverage**
   - No automated tests for validation logic, bus behavior under jitter/drop, or safety-mode transitions; hazard analyses and fault-injection simulations are absent.

## Recommendations to improve
- **Formalize signal schemas**: Define structured contracts (units, ranges, plausibility checks, freshness counters, diagnostic categories) and enforce them in `validate()`; reject or downgrade outputs on violations. 【F:ecu.py†L24-L170】
- **Enhance bus semantics**: Add priority queues, deadlines, watchdog timeouts, and explicit acknowledgement/failure paths; expose delivery/expiry into SafetyManager so degraded behavior triggers when communication falters. 【F:ecu.py†L298-L371】【F:ecu.py†L381-L416】
- **Expand safety strategy**: Introduce multi-level degraded modes (e.g., torque-limit, ramp-to-stop), configurable reset/latch policies, and logging of fault provenance; integrate perception/actuation health signals and ABS self-checks. 【F:ecu.py†L381-L416】
- **Secure calibrations**: Validate against a formal schema, verify signatures or checksums at startup, and select datasets by environment with version tracking to prevent misconfiguration. 【F:calibrations.py†L41-L63】
- **Strengthen observability**: Emit structured telemetry (JSON/CSV) for each message hop, safety decision, and actuator command with timestamps; track bus backlog, missed deadlines, and fault rates for runtime monitoring. 【F:ecu.py†L298-L371】【F:ecu.py†L381-L416】
- **Add verification tooling**: Build unit/integration tests for signal validation, bus jitter/drop behavior, and safety-mode transitions; incorporate fault injection and scenario-based tests (e.g., stale perception, ABS failure) to measure reaction times and ensure ASIL-aligned coverage.
