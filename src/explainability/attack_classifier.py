"""Rule-based attack type classification.

Classifies each detected anomaly window into one of:
  - sensor_manipulation: continuous sensor readings were altered
  - actuator_control: actuator state was changed illegitimately
  - sensor_malfunction: sensor is frozen or producing spikes
  - unknown: no clear pattern

Uses feature attribution from explainer.py to determine the type.
"""

from src.explainability.explainer import get_stage, STAGE_MAP


ATTACK_TYPES = {
    "sensor_manipulation": "Continuous sensor readings deviate from normal without "
                           "corresponding actuator changes",
    "actuator_control": "Actuator state changed, causing physical contradiction "
                        "(e.g., pump ON but no flow)",
    "sensor_malfunction": "Sensor is frozen (zero variance) or producing abnormal "
                          "spikes/noise",
    "unknown": "Anomaly detected but no clear attack pattern identified",
}


def classify_attack(explanation: dict) -> dict:
    """Classify an attack window based on its feature attribution.

    Returns dict with attack_type, involved_sensors, evidence.
    """
    category_totals = explanation["category_totals"]
    sensor_totals = explanation["sensor_totals"]
    top_features = explanation["top_features"]

    total_error = category_totals.sum()
    if total_error == 0:
        return _build_result("unknown", [], "No significant reconstruction error")

    category_fractions = category_totals / total_error

    contradiction_share = category_fractions.get("contradiction", 0)
    constant_share = category_fractions.get("constant_change", 0)
    frozen_share = category_fractions.get("frozen", 0)
    spike_share = category_fractions.get("spike_jump", 0)
    switching_share = category_fractions.get("switching", 0)
    deviation_share = category_fractions.get("deviation", 0)
    rolling_share = category_fractions.get("rolling", 0)

    actuator_signal = contradiction_share + constant_share + switching_share
    malfunction_signal = frozen_share + spike_share

    involved = _extract_involved_sensors(sensor_totals, total_error)

    if actuator_signal > 0.3:
        evidence = _build_evidence_actuator(top_features, category_fractions)
        return _build_result("actuator_control", involved, evidence)

    if malfunction_signal > 0.3:
        evidence = _build_evidence_malfunction(top_features, category_fractions)
        return _build_result("sensor_malfunction", involved, evidence)

    if deviation_share + rolling_share > 0.3:
        evidence = _build_evidence_manipulation(top_features, category_fractions)
        return _build_result("sensor_manipulation", involved, evidence)

    top_cat = category_totals.index[0] if len(category_totals) > 0 else "unknown"
    evidence = f"Dominant feature category: {top_cat} ({category_fractions.get(top_cat, 0):.0%})"
    attack_type = _fallback_type(top_cat)
    return _build_result(attack_type, involved, evidence)


def _extract_involved_sensors(
    sensor_totals, total_error, threshold: float = 0.05,
) -> list[dict]:
    """Extract sensors with significant contribution."""
    involved = []
    for sensor, error in sensor_totals.items():
        fraction = error / total_error
        if fraction < threshold:
            continue
        involved.append({
            "sensor": sensor,
            "stage": get_stage(sensor),
            "contribution": round(fraction, 3),
        })
    return involved


def _build_result(
    attack_type: str, involved: list[dict], evidence: str,
) -> dict:
    return {
        "attack_type": attack_type,
        "description": ATTACK_TYPES[attack_type],
        "involved_sensors": involved,
        "evidence": evidence,
    }


def _build_evidence_actuator(top_features, fractions) -> str:
    contra_feats = top_features[
        top_features["category"].isin(["contradiction", "constant_change", "switching"])
    ]
    if len(contra_feats) == 0:
        return "Actuator-related features dominate reconstruction error"
    top = contra_feats.iloc[0]
    return (
        f"Physical contradiction detected: {top['feature']} "
        f"(sensor {top['sensor']}, stage {top['stage']}). "
        f"Actuator features account for "
        f"{fractions.get('contradiction', 0) + fractions.get('constant_change', 0):.0%} "
        f"of total error."
    )


def _build_evidence_malfunction(top_features, fractions) -> str:
    mal_feats = top_features[
        top_features["category"].isin(["frozen", "spike_jump"])
    ]
    if len(mal_feats) == 0:
        return "Frozen/spike features dominate reconstruction error"
    top = mal_feats.iloc[0]
    return (
        f"Sensor malfunction: {top['feature']} "
        f"(sensor {top['sensor']}, stage {top['stage']}). "
        f"Frozen={fractions.get('frozen', 0):.0%}, "
        f"spike={fractions.get('spike_jump', 0):.0%} of total error."
    )


def _build_evidence_manipulation(top_features, fractions) -> str:
    manip_feats = top_features[
        top_features["category"].isin(["deviation", "rolling", "rate_of_change"])
    ]
    if len(manip_feats) == 0:
        return "Sensor value deviation dominates reconstruction error"
    top = manip_feats.iloc[0]
    return (
        f"Sensor manipulation: {top['feature']} "
        f"(sensor {top['sensor']}, stage {top['stage']}). "
        f"Deviation={fractions.get('deviation', 0):.0%}, "
        f"rolling={fractions.get('rolling', 0):.0%} of total error."
    )


def _fallback_type(category: str) -> str:
    if category in ("contradiction", "constant_change", "switching"):
        return "actuator_control"
    if category in ("frozen", "spike_jump"):
        return "sensor_malfunction"
    if category in ("deviation", "rolling", "rate_of_change", "raw_sensor"):
        return "sensor_manipulation"
    return "unknown"
