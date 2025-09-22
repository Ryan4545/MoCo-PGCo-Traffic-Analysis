#!/usr/bin/env python3
"""
Data Limitations Analysis for MoCo ASE Safety Study

This script analyzes the current data limitations and suggests alternative approaches
for studying automated speed enforcement effectiveness.
"""

import pandas as pd
import numpy as np
from analysis_pipeline import load_moco_ase, load_moco_crashes, crashes_within_radius

def analyze_data_limitations():
    """Analyze what we can and cannot do with current data"""
    
    print("=" * 60)
    print("MONTGOMERY COUNTY ASE SAFETY ANALYSIS - DATA LIMITATIONS")
    print("=" * 60)
    
    # Load data
    ase_data = load_moco_ase()
    crash_data = load_moco_crashes()
    
    print("\n1. CURRENT DATA AVAILABILITY:")
    print(f"   • ASE sites: {ase_data['site_id'].nunique()} locations")
    print(f"   • Crash records: {len(crash_data):,} total crashes")
    print(f"   • Crash date range: {crash_data['crash_datetime'].min()} to {crash_data['crash_datetime'].max()}")
    print(f"   • Citation data: Only 2024 quarterly totals available")
    
    print("\n2. CRITICAL MISSING DATA:")
    print("   ❌ Historical camera installation dates")
    print("   ❌ Site-specific activation timelines") 
    print("   ❌ Pre-installation baseline data")
    print("   ❌ Camera deployment phases (2007, 2011, etc.)")
    
    print("\n3. WHAT THIS MEANS FOR ANALYSIS:")
    print("   ❌ Cannot do proper before/after analysis")
    print("   ❌ Cannot measure safety impact of camera installation")
    print("   ❌ Cannot control for selection bias in camera placement")
    print("   ❌ Cannot account for different installation periods")
    
    print("\n4. ALTERNATIVE ANALYSES POSSIBLE:")
    print("   ✅ Geographic analysis: Compare crash rates near vs far from cameras")
    print("   ✅ Cross-sectional analysis: High vs low citation sites")
    print("   ✅ Temporal analysis: Crash trends over time (without activation dates)")
    print("   ✅ Spatial correlation: Citation intensity vs crash patterns")
    
    # Perform geographic analysis instead
    print("\n5. RECOMMENDED ALTERNATIVE APPROACH:")
    print("   → Geographic Proximity Analysis")
    print("   → Compare crash rates within 300m vs 300-600m from cameras")
    print("   → Control for road type, traffic volume, etc.")
    
    return ase_data, crash_data

def geographic_proximity_analysis():
    """Alternative analysis using geographic proximity instead of time"""
    
    print("\n" + "=" * 60)
    print("GEOGRAPHIC PROXIMITY ANALYSIS (Alternative Approach)")
    print("=" * 60)
    
    # Load data
    ase_data = load_moco_ase()
    crash_data = load_moco_crashes()
    
    # Create camera locations
    cams_geo = ase_data.groupby(["site_id","lat","lon"])["citations"].sum().reset_index()
    
    # Find crashes within different distance bands
    print("\nAnalyzing crashes at different distances from cameras...")
    
    # Within 300m (current analysis)
    crashes_300m = crashes_within_radius(crash_data, cams_geo, radius_m=300.0)
    
    # Within 600m (control group)
    crashes_600m = crashes_within_radius(crash_data, cams_geo, radius_m=600.0)
    
    # Calculate rates
    injury_rate_300m = crashes_300m['injury_crash'].mean()
    injury_rate_600m = crashes_600m['injury_crash'].mean()
    
    print(f"\nRESULTS:")
    print(f"   • Crashes within 300m of cameras: {len(crashes_300m):,}")
    print(f"   • Crashes within 600m of cameras: {len(crashes_600m):,}")
    print(f"   • Injury rate within 300m: {injury_rate_300m:.1%}")
    print(f"   • Injury rate within 600m: {injury_rate_600m:.1%}")
    
    if injury_rate_300m < injury_rate_600m:
        print(f"   → Cameras appear to reduce injury crashes by {(injury_rate_600m - injury_rate_300m)*100:.1f} percentage points")
    else:
        print(f"   → No clear safety benefit detected in proximity analysis")
    
    return crashes_300m, crashes_600m

def suggest_data_sources():
    """Suggest where to find missing activation data"""
    
    print("\n" + "=" * 60)
    print("SUGGESTED DATA SOURCES FOR ACTIVATION DATES")
    print("=" * 60)
    
    print("\n1. MONTGOMERY COUNTY GOVERNMENT:")
    print("   • Department of Transportation")
    print("   • Police Department (traffic enforcement)")
    print("   • Public records requests")
    print("   • County council meeting minutes")
    
    print("\n2. PRINCE GEORGE'S COUNTY (if applicable):")
    print("   • Similar government sources")
    print("   • Cross-reference with MoCo data")
    
    print("\n3. HISTORICAL RESEARCH:")
    print("   • News articles about camera installations")
    print("   • Government press releases")
    print("   • Traffic safety reports")
    print("   • Academic studies on ASE implementation")
    
    print("\n4. DATA REQUEST TEMPLATE:")
    print("   'Request for historical data on automated speed enforcement")
    print("   camera installation dates, locations, and activation timelines")
    print("   for Montgomery County from 2007-present'")
    
    print("\n5. ALTERNATIVE RESEARCH DESIGN:")
    print("   • Focus on geographic proximity analysis")
    print("   • Compare high-enforcement vs low-enforcement areas")
    print("   • Use traffic volume data to control for exposure")
    print("   • Analyze seasonal and temporal patterns")

if __name__ == "__main__":
    # Run the analysis
    ase_data, crash_data = analyze_data_limitations()
    crashes_300m, crashes_600m = geographic_proximity_analysis()
    suggest_data_sources()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Without historical activation dates, a proper before/after analysis")
    print("is not possible. Consider geographic proximity analysis as an")
    print("alternative approach, or obtain historical installation data.")
    print("=" * 60)
