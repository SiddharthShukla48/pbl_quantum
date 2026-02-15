"""
Room Assignment (Classical Post-Processing)
============================================

After QAOA assigns exams to time slots, this script assigns exams to rooms.

Method: First-Fit Decreasing Bin Packing
-----------------------------------------
For each time slot:
1. Sort exams by enrollment (largest first)
2. Assign each exam to first room with enough capacity
3. If no room fits, flag as overflow

This is CLASSICAL (not quantum) and runs instantly.

Author: For quantum graph coloring exam scheduling research
Date: February 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys


def get_latest_run_dir(output_base='./output'):
    """Get the latest run directory"""
    latest_file = Path(output_base) / 'latest_run.txt'
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            return Path(f.read().strip())
    return None


def load_coloring_solution(solutions_dir, dataset, K):
    """
    Load the QAOA coloring solution
    
    Args:
        solutions_dir: Solutions directory path
        dataset: Dataset size name
        K: Number of colors
        
    Returns:
        dict {exam_id: slot}
    """
    solution_path = Path(solutions_dir) / f'qaoa_solution_{dataset}_K{K}.json'
    
    if not solution_path.exists():
        print(f"⚠ No saved solution found at {solution_path}")
        print("Run 04_solve_qaoa.py first and save the coloring solution.")
        return None
    
    with open(solution_path, 'r') as f:
        # Solution saved as {"0": 1, "1": 0, ...}
        coloring_str = json.load(f)
        # Convert keys back to integers
        coloring = {int(k): v for k, v in coloring_str.items()}
    
    return coloring


def assign_rooms_to_exams(coloring, courses_df, rooms_df):
    """
    Assign exams to rooms using First-Fit Decreasing
    
    Args:
        coloring: dict {exam_id: slot}
        courses_df: DataFrame with course info and enrollments
        rooms_df: DataFrame with room info and capacities
        
    Returns:
        dict {(exam_id, slot): room_id} or None if assignment fails
    """
    print("\n" + "="*60)
    print("ROOM ASSIGNMENT (First-Fit Decreasing)")
    print("="*60)
    
    # Group exams by slot
    slots = {}
    for exam_id, slot in coloring.items():
        if slot not in slots:
            slots[slot] = []
        slots[slot].append(exam_id)
    
    print(f"\nTime slots used: {len(slots)}")
    for slot, exams in sorted(slots.items()):
        print(f"  Slot {slot}: {len(exams)} exams")
    
    # Sort rooms by capacity (largest first, for efficiency)
    rooms_sorted = rooms_df.sort_values('capacity', ascending=False)
    
    # Assignment result
    assignment = {}  # {(exam_id, slot): room_id}
    overflow = []    # Exams that don't fit
    
    # For each slot, assign exams to rooms
    for slot in sorted(slots.keys()):
        exams_in_slot = slots[slot]
        
        print(f"\n--- Slot {slot} ---")
        
        # Sort exams by enrollment (largest first)
        exams_sorted = sorted(
            exams_in_slot,
            key=lambda e: courses_df.iloc[e]['enrollment'],
            reverse=True
        )
        
        # Track room usage in this slot
        room_usage = {}  # {room_id: remaining_capacity}
        for _, room in rooms_sorted.iterrows():
            room_usage[room['room_id']] = room['capacity']
        
        # Assign each exam
        for exam_id in exams_sorted:
            enrollment = courses_df.iloc[exam_id]['enrollment']
            course_code = courses_df.iloc[exam_id]['course_code']
            
            assigned = False
            
            # Try to fit in existing rooms
            for room_id in room_usage:
                if room_usage[room_id] >= enrollment:
                    # Assign to this room
                    assignment[(exam_id, slot)] = room_id
                    room_usage[room_id] -= enrollment
                    
                    room_name = rooms_sorted[rooms_sorted['room_id'] == room_id].iloc[0]['room_name']
                    print(f"  {course_code} ({enrollment} students) → {room_name}")
                    
                    assigned = True
                    break
            
            if not assigned:
                overflow.append((exam_id, slot, enrollment))
                print(f"  ⚠ {course_code} ({enrollment} students) → NO ROOM AVAILABLE")
    
    # Summary
    print("\n" + "="*60)
    print("ASSIGNMENT SUMMARY")
    print("="*60)
    print(f"Total exams:        {len(coloring)}")
    print(f"Successfully assigned: {len(assignment)}")
    print(f"Overflow (no room):   {len(overflow)}")
    
    if overflow:
        print("\n⚠ OVERFLOW EXAMS:")
        for exam_id, slot, enrollment in overflow:
            course_code = courses_df.iloc[exam_id]['course_code']
            print(f"  {course_code} (Slot {slot}, {enrollment} students)")
        print("\nSuggestions:")
        print("  1. Add more rooms")
        print("  2. Split large exams across multiple rooms")
        print("  3. Use more time slots")
    
    return assignment, overflow


def generate_timetable(assignment, courses_df, rooms_df):
    """
    Generate final timetable in readable format
    
    Args:
        assignment: dict {(exam_id, slot): room_id}
        courses_df: DataFrame with courses
        rooms_df: DataFrame with rooms
        
    Returns:
        DataFrame with timetable
    """
    rows = []
    
    for (exam_id, slot), room_id in assignment.items():
        course_code = courses_df.iloc[exam_id]['course_code']
        year = courses_df.iloc[exam_id]['year']
        enrollment = courses_df.iloc[exam_id]['enrollment']
        
        room_name = rooms_df[rooms_df['room_id'] == room_id].iloc[0]['room_name']
        room_capacity = rooms_df[rooms_df['room_id'] == room_id].iloc[0]['capacity']
        
        rows.append({
            'Slot': slot,
            'Course': course_code,
            'Year': year,
            'Enrollment': enrollment,
            'Room': room_name,
            'Room_Capacity': room_capacity,
            'Utilization': f"{enrollment/room_capacity*100:.1f}%"
        })
    
    timetable = pd.DataFrame(rows)
    timetable = timetable.sort_values(['Slot', 'Course'])
    
    return timetable


def save_timetable(timetable, output_path):
    """Save timetable to CSV"""
    timetable.to_csv(output_path, index=False)
    print(f"\n✓ Timetable saved to {output_path}")


def print_timetable_pretty(timetable):
    """Print formatted timetable"""
    print("\n" + "="*80)
    print("FINAL EXAM TIMETABLE")
    print("="*80)
    
    for slot in sorted(timetable['Slot'].unique()):
        print(f"\n{'='*80}")
        print(f"TIME SLOT {slot}")
        print(f"{'='*80}")
        
        slot_exams = timetable[timetable['Slot'] == slot]
        
        for _, row in slot_exams.iterrows():
            print(f"{row['Course']:12} | Year {row['Year']} | "
                  f"{row['Enrollment']:3} students | "
                  f"{row['Room']:12} (capacity {row['Room_Capacity']}) | "
                  f"Util: {row['Utilization']}")


def main():
    """Main room assignment workflow"""
    import sys
    
    print("="*60)
    print("ROOM ASSIGNMENT FOR EXAM TIMETABLING")
    print("="*60)
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
    else:
        dataset = 'tiny'
    
    if len(sys.argv) > 2:
        K = int(sys.argv[2])
    else:
        K_defaults = {'tiny': 3, 'small': 4, 'medium': 5}
        K = K_defaults.get(dataset, 3)
    
    print(f"\n📊 Dataset: {dataset.upper()}")
    print(f"🎨 Colors (K): {K}\n")
    
    # Get latest run directory
    run_dir = get_latest_run_dir()
    if run_dir is None:
        print("⚠️ No run directory found. Please run 01_generate_dataset.py first.")
        return
    
    datasets_dir = run_dir / 'datasets'
    solutions_dir = run_dir / 'solutions'
    timetables_dir = run_dir / 'timetables'
    timetables_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Using run directory: {run_dir}")
    print(f"📁 Timetables will be saved to: {timetables_dir}\n")
    
    # Load data
    data_dir = datasets_dir / f'exam_data_{dataset}'
    
    if not data_dir.exists():
        print(f"\n⚠ Error: {data_dir} not found!")
        return
    
    courses_df = pd.read_csv(data_dir / 'courses.csv')
    rooms_df = pd.read_csv(data_dir / 'rooms.csv')
    
    print(f"\nDataset: {dataset.upper()}")
    print(f"Courses: {len(courses_df)}")
    print(f"Rooms: {len(rooms_df)}")
    
    # Load QAOA coloring solution
    coloring = load_coloring_solution(solutions_dir, dataset, K)
    
    if coloring is None:
        print("\n⚠ No QAOA solution found. Creating dummy solution for demonstration.")
        print("To use real QAOA solution, run 04_solve_qaoa.py first.")
        # Dummy: assign exams to slots in round-robin fashion
        coloring = {i: i % K for i in range(len(courses_df))}
    else:
        print(f"✓ Loaded QAOA solution with {len(coloring)} exam assignments")
    
    # Assign rooms
    assignment, overflow = assign_rooms_to_exams(coloring, courses_df, rooms_df)
    
    if assignment:
        # Generate timetable
        timetable = generate_timetable(assignment, courses_df, rooms_df)
        
        # Print
        print_timetable_pretty(timetable)
        
        # Save
        output_file = timetables_dir / f'final_timetable_{dataset}_K{K}.csv'
        save_timetable(timetable, output_file)
        
        # Statistics
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        
        avg_util = timetable['Enrollment'].sum() / timetable['Room_Capacity'].sum() * 100
        print(f"Average room utilization: {avg_util:.1f}%")
        
        slots_used = len(timetable['Slot'].unique())
        rooms_used = len(timetable['Room'].unique())
        print(f"Time slots used: {slots_used}/{K}")
        print(f"Rooms used: {rooms_used}/{len(rooms_df)}")


if __name__ == '__main__':
    main()
