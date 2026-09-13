import re
from datetime import datetime

def validate_mrz_checksum(mrz_string):
    """
    Validates MRZ checksums.
    A simplified version for the prototype.
    """
    if not mrz_string:
        return False
        
    # Example logic: in a real system we'd calculate weights (7,3,1)
    # Here we'll just check if it matches basic MRZ formatting
    # Assuming MRZ contains '<' and alphanumeric characters
    has_fillers = '<' in mrz_string
    has_alphanum = any(c.isalnum() for c in mrz_string)
    
    return has_fillers and has_alphanum

def check_date_consistency(dob_str, expiry_str):
    """
    Check logical consistency: DOB < Expiry Date
    """
    try:
        # Assuming format YYYY-MM-DD or DD/MM/YYYY for parsed strings
        # In a real system, we'd parse the specific MRZ YYMMDD format
        dob = pd.to_datetime(dob_str)
        expiry = pd.to_datetime(expiry_str)
        return dob < expiry
    except:
        return None # Not available or parsing failed

def run_document_rules(extracted_data):
    """
    Run structural and logical rules on extracted OCR/MRZ data.
    """
    rules_results = {}
    
    mrz = extracted_data.get("mrz", "")
    if mrz:
        rules_results["MRZ_Checksum"] = "PASS" if validate_mrz_checksum(mrz) else "FAIL"
    else:
        rules_results["MRZ_Checksum"] = "NOT_AVAILABLE"
        
    visual_name = extracted_data.get("visual_name", "").upper()
    mrz_name = extracted_data.get("mrz_name", "").upper()
    if visual_name and mrz_name:
        # fuzzy match or strict
        rules_results["Name_Consistency"] = "PASS" if visual_name in mrz_name or mrz_name in visual_name else "FAIL"
    else:
        rules_results["Name_Consistency"] = "NOT_AVAILABLE"
        
    return rules_results

def calculate_risk_score(cnn_prob, ela_stats, rules_results):
    """
    Calculate an explainable risk score (0-100).
    Higher is more risky.
    """
    score = 0
    reasons = []
    
    # 1. CNN Evidence (weight: 40)
    score += cnn_prob * 40
    if cnn_prob > 0.7:
        reasons.append("Elevated CNN forgery probability.")
        
    # 2. Forensics (weight: 30)
    # ELA anomaly threshold example
    if ela_stats.get("ela_max", 0) > 100: 
        score += 30
        reasons.append("ELA anomaly detected in image regions.")
        
    # 3. Rules (weight: 30)
    rule_failures = 0
    for rule, res in rules_results.items():
        if res == "FAIL":
            rule_failures += 1
            reasons.append(f"Rule failed: {rule.replace('_', ' ')}")
            
    score += min(30, rule_failures * 15)
    
    # Cap at 100
    score = min(100, int(score))
    
    status = "LOW RISK"
    if score >= 80:
        status = "CRITICAL RISK"
    elif score >= 60:
        status = "HIGH RISK"
    elif score >= 30:
        status = "MEDIUM RISK"
        
    return {
        "score": score,
        "status": status,
        "reasons": reasons
    }
