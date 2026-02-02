"""
Data Loader for Exam Scheduling QUBO Framework
Loads, validates, and processes the generated datasets

Author: Created for exam scheduling QUBO research
Date: February 2026
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class ExamSchedulingData:
    """Container for all exam scheduling data"""
    courses: pd.DataFrame
    students: pd.DataFrame
    enrollments: pd.DataFrame
    rooms: pd.DataFrame
    faculty: pd.DataFrame
    conflict_matrix: np.ndarray
    faculty_preferences: pd.DataFrame
    room_availability: pd.DataFrame
    metadata: Dict


class ExamSchedulingDataLoader:
    """Load and manage exam scheduling datasets"""
    
    def __init__(self, data_dir: str):
        """
        Initialize loader
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load(self) -> ExamSchedulingData:
        """Load all datasets from CSV files"""
        
        print(f"Loading data from {self.data_dir}...")
        
        # Load CSV files
        courses = pd.read_csv(self.data_dir / 'courses.csv')
        students = pd.read_csv(self.data_dir / 'students.csv')
        enrollments = pd.read_csv(self.data_dir / 'enrollments.csv')
        rooms = pd.read_csv(self.data_dir / 'rooms.csv')
        faculty = pd.read_csv(self.data_dir / 'faculty.csv')
        conflict_matrix = pd.read_csv(self.data_dir / 'conflict_matrix.csv', index_col=0)
        faculty_preferences = pd.read_csv(self.data_dir / 'faculty_preferences.csv')
        room_availability = pd.read_csv(self.data_dir / 'room_availability.csv')
        
        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Convert conflict matrix to numpy array
        conflict_array = conflict_matrix.values.astype(int)
        
        # Create data object
        data = ExamSchedulingData(
            courses=courses,
            students=students,
            enrollments=enrollments,
            rooms=rooms,
            faculty=faculty,
            conflict_matrix=conflict_array,
            faculty_preferences=faculty_preferences,
            room_availability=room_availability,
            metadata=metadata
        )
        
        print(f"✓ Loaded {len(courses)} courses")
        print(f"✓ Loaded {len(students)} students")
        print(f"✓ Loaded {len(enrollments)} enrollments")
        print(f"✓ Loaded {len(rooms)} rooms")
        print(f"✓ Loaded {len(faculty)} faculty members")
        
        return data
    
    def print_statistics(self, data: ExamSchedulingData):
        """Print comprehensive statistics about the loaded data"""
        
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        
        # Metadata
        print(f"\nDataset Configuration:")
        print(f"  Courses:              {data.metadata['num_courses']}")
        print(f"  Students:             {data.metadata['num_students']}")
        print(f"  Rooms:                {data.metadata['num_rooms']}")
        print(f"  Faculty:              {data.metadata['num_faculty']}")
        print(f"  Exam Days:            {data.metadata['num_days']}")
        print(f"  Timeslots per Day:    {data.metadata['num_timeslots']}")
        
        # Course statistics
        print(f"\nCourse Information:")
        enrollments_per_course = data.enrollments.groupby('course_id').size()
        print(f"  Total Enrollments:    {len(data.enrollments)}")
        print(f"  Avg Enrollment:       {enrollments_per_course.mean():.1f}")
        print(f"  Min Enrollment:       {enrollments_per_course.min()}")
        print(f"  Max Enrollment:       {enrollments_per_course.max()}")
        
        # Conflict analysis
        print(f"\nConflict Analysis:")
        conflict_pairs = np.where(data.conflict_matrix > 0)
        num_conflict_pairs = len(conflict_pairs[0])
        max_conflict_size = data.conflict_matrix.max()
        print(f"  Conflicting Pairs:    {num_conflict_pairs}")
        print(f"  Max Conflict Size:    {max_conflict_size}")
        
        # Room analysis
        print(f"\nRoom Information:")
        print(f"  Total Rooms:          {len(data.rooms)}")
        print(f"  Total Capacity:       {data.rooms['capacity'].sum()}")
        print(f"  Avg Room Capacity:    {data.rooms['capacity'].mean():.1f}")
        
        print(f"\n  By Room Type:")
        for room_type in data.rooms['room_type'].unique():
            rooms_of_type = data.rooms[data.rooms['room_type'] == room_type]
            count = len(rooms_of_type)
            avg_cap = rooms_of_type['capacity'].mean()
            print(f"    {room_type:15}: {count} rooms (avg capacity: {avg_cap:.0f})")
        
        # Faculty analysis
        print(f"\nFaculty Information:")
        print(f"  Total Faculty:        {len(data.faculty)}")
        print(f"  Total Duty Quota:     {data.faculty['invigilator_quota'].sum()}")
        
        print(f"\n  By Rank:")
        for rank in sorted(data.faculty['rank'].unique()):
            faculty_of_rank = data.faculty[data.faculty['rank'] == rank]
            count = len(faculty_of_rank)
            quota_per_person = faculty_of_rank['invigilator_quota'].iloc[0]
            total_quota = faculty_of_rank['invigilator_quota'].sum()
            print(f"    {rank:20}: {count} faculty (quota: {quota_per_person} each, total: {total_quota})")
        
        print("\n" + "="*70)


# Example usage
if __name__ == '__main__':
    import sys
    
    # Load a specific dataset
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = './exam_scheduling_data_small'
    
    try:
        loader = ExamSchedulingDataLoader(data_dir)
        data = loader.load()
        
        loader.print_statistics(data)
        
        print("\n✓ Data successfully loaded!")
        print("\nYou can now use this data to build your QUBO matrices.")
        print("Example:")
        print("  courses = data.courses")
        print("  conflict_matrix = data.conflict_matrix")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nMake sure to run 'python dataset-generator.py' first!")
        sys.exit(1)
