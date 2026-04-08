"""
Resource Estimation Agent
Estimates hospital resource requirements based on predicted at-risk patient count.
"""


def estimate_resources(predicted_at_risk: int) -> dict:
    """Calculate hospital resource requirements."""
    try:
        beds_needed = round(predicted_at_risk * 0.3)
        doctors_needed = round(beds_needed / 5)
        estimated_cost = beds_needed * 5000
        treatment_days = beds_needed * 2

        resources = {
            "predicted_patients": predicted_at_risk,
            "beds_needed": beds_needed,
            "doctors_needed": doctors_needed,
            "estimated_cost_usd": estimated_cost,
            "treatment_days": treatment_days,
        }

        print(f"[ResourceEstimationAgent] ✅ Resource estimates calculated:")
        for k, v in resources.items():
            label = k.replace("_", " ").title()
            print(f"  • {label}: {v:,}" if isinstance(v, int) else f"  • {label}: {v}")

        return resources

    except Exception as e:
        print(f"[ResourceEstimationAgent] ❌ Error: {e}")
        raise


if __name__ == "__main__":
    result = estimate_resources(50)
    print(result)
