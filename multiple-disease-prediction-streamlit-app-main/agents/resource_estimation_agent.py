"""
Resource Estimation Agent
Computes hospital resources based on prediction risk counts.
Innovation Feature.
"""

def estimate_resources(at_risk_count: int, disease_type: str = "diabetes"):
    """Calculate beds, doctors, cost, and time."""
    # Disease specific multipliers
    config = {
        "diabetes": {"beds": 0.3, "doctors_div": 5, "cost": 5000, "days": 2},
        "heart": {"beds": 0.8, "doctors_div": 3, "cost": 15000, "days": 5},
        "parkinsons": {"beds": 0.2, "doctors_div": 8, "cost": 8000, "days": 10}
    }

    c = config.get(disease_type, config["diabetes"])

    beds_needed = round(at_risk_count * c["beds"])
    doctors_needed = round(beds_needed / c["doctors_div"]) + (1 if beds_needed > 0 else 0)
    estimated_cost = beds_needed * c["cost"]
    treatment_days = beds_needed * c["days"]

    resources = {
        "at_risk": at_risk_count,
        "beds_needed": beds_needed,
        "doctors_needed": doctors_needed,
        "estimated_cost": estimated_cost,
        "treatment_days": treatment_days
    }

    print(f"[ResourceEstimationAgent] ✅ Resources estimated for {disease_type}")
    print(f"[ResourceEstimationAgent] Beds needed: {beds_needed}, Cost: ${estimated_cost:,}")

    return resources
