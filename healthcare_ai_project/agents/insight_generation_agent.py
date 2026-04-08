"""
Insight Generation Agent
Produces a human-readable healthcare analytics report.
"""

from datetime import datetime


def generate_insights(prediction_result: dict, resource_result: dict, accuracy: float) -> str:
    """Generate and print a structured insights report."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║          MediPredict AI — Healthcare Analytics Report            ║
╚══════════════════════════════════════════════════════════════════╝

  Generated at  : {timestamp}
  Model Accuracy: {accuracy * 100:.2f}%

┌──────────────────────────────────────────────────────────────────┐
│  PREDICTION SUMMARY                                              │
├──────────────────────────────────────────────────────────────────┤
  Total Patients Assessed  : {prediction_result['total_patients']:>8,}
  Predicted At-Risk        : {prediction_result['predicted_at_risk']:>8,}
  Risk Rate                : {prediction_result['positive_rate']:>7.1f}%

┌──────────────────────────────────────────────────────────────────┐
│  RESOURCE ESTIMATES                                              │
├──────────────────────────────────────────────────────────────────┤
  Beds Required            : {resource_result['beds_needed']:>8,}
  Doctors Required         : {resource_result['doctors_needed']:>8,}
  Estimated Cost (USD)     : ${resource_result['estimated_cost_usd']:>10,}
  Treatment Days           : {resource_result['treatment_days']:>8,}

┌──────────────────────────────────────────────────────────────────┐
│  RECOMMENDATIONS                                                 │
├──────────────────────────────────────────────────────────────────┤
  ✅ Allocate {resource_result['beds_needed']} beds for incoming high-risk patients.
  ✅ Schedule {resource_result['doctors_needed']} doctors for the estimated patient load.
  ✅ Budget approximately ${resource_result['estimated_cost_usd']:,} for treatment costs.
  ✅ Plan for {resource_result['treatment_days']} treatment days across all cases.

══════════════════════════════════════════════════════════════════
"""
        print(report)
        return report

    except Exception as e:
        print(f"[InsightGenerationAgent] ❌ Error: {e}")
        raise


if __name__ == "__main__":
    preds = {"total_patients": 154, "predicted_at_risk": 55, "positive_rate": 35.7}
    resources = {
        "predicted_patients": 55,
        "beds_needed": 17,
        "doctors_needed": 3,
        "estimated_cost_usd": 85000,
        "treatment_days": 34,
    }
    generate_insights(preds, resources, accuracy=0.76)
