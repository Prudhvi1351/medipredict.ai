"""
Insight Generation Agent
Generates human-readable healthcare conclusion reports.
"""
from datetime import datetime

def generate_report(prediction_data, resource_data, accuracy, disease_type="diabetes"):
    """Create a structured text report of the analytics results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║        Autonomous Healthcare Insights: {disease_type.upper():<18} ║
╚══════════════════════════════════════════════════════════════════╝

  Analysis Date   : {timestamp}
  Predictive Accuracy: {accuracy*100:.1f}%

  ── PREDICTION RESULTS ───────────────────────────────────────────
  Total Screened       : {prediction_data['total']}
  Predicted High-Risk  : {prediction_data['at_risk']}
  Risk Prevalence     : {(prediction_data['at_risk']/prediction_data['total'])*100:.1f}%

  ── RESOURCE PLANNING (Innovation) ───────────────────────────────
  Estimated Patients   : {resource_data['at_risk']}
  Hospital Beds Required: {resource_data['beds_needed']}
  Staffing Requirement : {resource_data['doctors_needed']} Doctors/Specialists
  Financial Impact    : ${resource_data['estimated_cost']:,} (USD)
  Operational Duration : {resource_data['treatment_days']} Plan Days

  ── CONCLUSIONS ──────────────────────────────────────────────────
  • Immediate allocation of {resource_data['beds_needed']} beds is recommended for pending cases.
  • Staff schedules should account for {resource_data['doctors_needed']} specialists.
  • Patient risk rate of {(prediction_data['at_risk']/prediction_data['total'])*100:.1f}% indicates a 
    {"significant" if prediction_data['at_risk']/prediction_data['total'] > 0.2 else "moderate"} healthcare burden.

══════════════════════════════════════════════════════════════════
"""
    print(f"[InsightGenerationAgent] ✅ Report generated for {disease_type}")
    print(report)
    return report
