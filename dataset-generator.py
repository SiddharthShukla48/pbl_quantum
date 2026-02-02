"""
Sample Dataset Generator for Exam Scheduling QUBO Framework
Generates realistic exam scheduling data for testing and prototyping

Author: Created for exam scheduling QUBO research
Date: February 2026
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple
import json
from pathlib import Path

class ExamSchedulingDataset:
    """Generate realistic exam scheduling datasets of various sizes"""
    
    def __init__(self, 
                 num_courses=10,
                 num_students=100,
                 num_rooms=5,
                 num_faculty=8,
                 num_days=5,
                 num_timeslots=2,
                 random_seed=42):
        """
        Initialize dataset generator
        
        Args:
            num_courses: Number of courses to schedule exams for
            num_students: Total number of students
            num_rooms: Number of available rooms
            num_faculty: Number of faculty members
            num_days: Number of exam days
            num_timeslots: Timeslots per day (typically 2: morning, afternoon)
            random_seed: For reproducibility
        """
        self.num_courses = num_courses
        self.num_students = num_students
        self.num_rooms = num_rooms
        self.num_faculty = num_faculty
        self.num_days = num_days
        self.num_timeslots = num_timeslots
        self.total_timeslots = num_days * num_timeslots
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Store generated data
        self.courses = None
        self.students = None
        self.enrollments = None
        self.rooms = None
        self.faculty = None
        self.conflict_matrix = None
        
    def generate_courses(self) -> pd.DataFrame:
        """Generate course data with realistic enrollments"""
        courses = []
        course_codes = ['CS101', 'CS102', 'CS201', 'CS202', 'CS301', 
                       'MATH101', 'MATH102', 'PHYS101', 'CHEM101', 'BIO101',
                       'ECO101', 'HIST101', 'ENG101', 'ENG102', 'PSYCH101']
        
        for i in range(self.num_courses):
            course_code = course_codes[i] if i < len(course_codes) else f'COURSE{i:03d}'
            
            # Enrollment: mostly 20-80, some large classes 100-150
            if i % 5 == 0:
                enrollment = np.random.randint(100, 150)  # Large lecture
            else:
                enrollment = np.random.randint(20, 80)  # Regular class
            
            # Exam duration: All exams same duration (1 slot = 3 hours is typical)
            # For QUBO purposes, we treat each exam as taking 1 timeslot
            # This makes the problem tractable and is realistic for scheduling
            duration = 1  # All exams take 1 timeslot (e.g., 3-hour block)
            
            # Room type requirement
            room_type = np.random.choice(['regular', 'computer_lab', 'regular'], 
                                        p=[0.7, 0.15, 0.15])
            
            # Faculty teaching the course
            faculty_id = np.random.randint(0, self.num_faculty)
            
            courses.append({
                'course_id': i,
                'course_code': course_code,
                'course_name': f'{course_code} - Course {i}',
                'enrollment': enrollment,
                'duration_hours': duration,
                'room_type_requirement': room_type,
                'instructor_faculty_id': faculty_id
            })
        
        self.courses = pd.DataFrame(courses)
        return self.courses
    
    def generate_students(self) -> pd.DataFrame:
        """Generate student data"""
        students = []
        for i in range(self.num_students):
            students.append({
                'student_id': i,
                'student_name': f'Student_{i:03d}',
                'department': 'Department_X',
                'year': np.random.randint(1, 4)
            })
        
        self.students = pd.DataFrame(students)
        return self.students
    
    def generate_enrollments(self) -> pd.DataFrame:
        """
        Generate enrollments (Student-Course mapping)
        Ensures realistic course enrollments and creates conflicts
        """
        enrollments = []
        
        for course_id, row in self.courses.iterrows():
            target_enrollment = row['enrollment']
            
            # Randomly select students for this course
            selected_students = np.random.choice(
                self.num_students,
                size=min(target_enrollment, self.num_students),
                replace=False
            )
            
            for student_id in selected_students:
                enrollments.append({
                    'student_id': student_id,
                    'course_id': course_id,
                    'course_code': self.courses.loc[course_id, 'course_code']
                })
        
        self.enrollments = pd.DataFrame(enrollments)
        return self.enrollments
    
    def generate_conflict_matrix(self) -> np.ndarray:
        """
        Generate conflict matrix: number of students taking both courses
        
        KEY CONCEPT: If a student is enrolled in BOTH course i and course j,
        then those courses CONFLICT (can't be scheduled at same time)
        
        The conflict_matrix[i,j] stores the NUMBER of students in both courses
        """
        conflict_matrix = np.zeros((self.num_courses, self.num_courses), dtype=int)
        
        # For each student, find all their courses
        for student_id in range(self.num_students):
            student_courses = self.enrollments[
                self.enrollments['student_id'] == student_id
            ]['course_id'].tolist()
            
            # For all pairs of courses this student is enrolled in,
            # increment the conflict count
            for i, course_i in enumerate(student_courses):
                for course_j in student_courses[i+1:]:
                    conflict_matrix[course_i, course_j] += 1
                    conflict_matrix[course_j, course_i] += 1  # Symmetric
        
        self.conflict_matrix = conflict_matrix
        return conflict_matrix
    
    def generate_rooms(self) -> pd.DataFrame:
        """Generate room data with capacities and types"""
        rooms = []
        room_types = ['regular', 'regular', 'regular', 'computer_lab', 'computer_lab']
        
        for i in range(self.num_rooms):
            room_type = room_types[i] if i < len(room_types) else 'regular'
            
            if room_type == 'regular':
                if i % 4 == 0:
                    capacity = 100  # Large lecture hall
                else:
                    capacity = 30   # Standard classroom
            else:  # computer_lab
                capacity = 40
            
            rooms.append({
                'room_id': i,
                'room_number': f'ROOM{i:02d}',
                'capacity': capacity,
                'room_type': room_type,
                'building': f'Building_{chr(65 + (i // 3))}',
                'floor': (i % 3) + 1
            })
        
        self.rooms = pd.DataFrame(rooms)
        return self.rooms
    
    def generate_faculty(self) -> pd.DataFrame:
        """
        Generate faculty data with ranks and duty quotas
        
        KEY CONCEPT: Faculty hierarchy determines invigilation duties:
        - Professor: 3 duties (senior, fewer duties)
        - Associate Professor: 4 duties
        - Assistant Professor: 5 duties (junior, more duties)
        """
        faculty = []
        
        # Distribute faculty by rank
        num_profs = max(1, self.num_faculty // 6)
        num_assoc = max(1, self.num_faculty // 3)
        num_asst = self.num_faculty - num_profs - num_assoc
        
        for rank_type, (rank_name, quota) in enumerate([
            ('Professor', 3),
            ('Associate Professor', 4),
            ('Assistant Professor', 5)
        ]):
            if rank_type == 0:
                count = num_profs
            elif rank_type == 1:
                count = num_assoc
            else:
                count = num_asst
            
            for i in range(count):
                faculty_id = len(faculty)
                faculty.append({
                    'faculty_id': faculty_id,
                    'faculty_name': f'{rank_name.replace(" ", "_")}{i:02d}',
                    'rank': rank_name,
                    'department': 'Department_X',
                    'email': f'{rank_name.replace(" ", "").lower()}{i}@university.edu',
                    'invigilator_quota': quota
                })
        
        self.faculty = pd.DataFrame(faculty)
        return self.faculty
    
    def generate_faculty_preferences(self) -> pd.DataFrame:
        """
        Generate faculty preferences for timeslots
        0 = available, 1 = low preference, 10 = unavailable
        """
        preferences = []
        
        for faculty_id in range(self.num_faculty):
            for day in range(self.num_days):
                for timeslot in range(self.num_timeslots):
                    # Most faculty available (60%), some low pref (20%), few unavailable (2%)
                    pref = np.random.choice([0, 0, 0, 1, 10], p=[0.6, 0.2, 0.1, 0.08, 0.02])
                    
                    preferences.append({
                        'faculty_id': faculty_id,
                        'day': day,
                        'timeslot': timeslot,
                        'preference_level': pref
                    })
        
        return pd.DataFrame(preferences)
    
    def generate_room_availability(self) -> pd.DataFrame:
        """
        Generate room availability per timeslot
        Most rooms available (95%), some unavailable for maintenance
        """
        availability = []
        
        for room_id in range(self.num_rooms):
            for day in range(self.num_days):
                for timeslot in range(self.num_timeslots):
                    available = np.random.choice([1, 0], p=[0.95, 0.05])
                    
                    availability.append({
                        'room_id': room_id,
                        'day': day,
                        'timeslot': timeslot,
                        'available': available
                    })
        
        return pd.DataFrame(availability)
    
    def generate_all(self) -> Dict:
        """Generate complete dataset and return as dictionary"""
        print("Generating dataset...")
        
        datasets = {
            'courses': self.generate_courses(),
            'students': self.generate_students(),
            'enrollments': self.generate_enrollments(),
            'rooms': self.generate_rooms(),
            'faculty': self.generate_faculty(),
            'conflict_matrix': pd.DataFrame(
                self.generate_conflict_matrix(),
                index=[f'Course_{i}' for i in range(self.num_courses)],
                columns=[f'Course_{i}' for i in range(self.num_courses)]
            ),
            'faculty_preferences': self.generate_faculty_preferences(),
            'room_availability': self.generate_room_availability()
        }
        
        print(f"✓ Generated {len(datasets['courses'])} courses")
        print(f"✓ Generated {len(datasets['students'])} students")
        print(f"✓ Generated {len(datasets['enrollments'])} enrollments")
        print(f"✓ Generated {len(datasets['rooms'])} rooms")
        print(f"✓ Generated {len(datasets['faculty'])} faculty members")
        print(f"✓ Generated conflict matrix ({self.num_courses}x{self.num_courses})")
        
        return datasets
    
    def save_to_csv(self, output_dir: str = './exam_scheduling_data'):
        """Save all datasets to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        datasets = self.generate_all()
        
        print(f"\nSaving to {output_dir}...")
        
        datasets['courses'].to_csv(output_path / 'courses.csv', index=False)
        datasets['students'].to_csv(output_path / 'students.csv', index=False)
        datasets['enrollments'].to_csv(output_path / 'enrollments.csv', index=False)
        datasets['rooms'].to_csv(output_path / 'rooms.csv', index=False)
        datasets['faculty'].to_csv(output_path / 'faculty.csv', index=False)
        datasets['conflict_matrix'].to_csv(output_path / 'conflict_matrix.csv')
        datasets['faculty_preferences'].to_csv(output_path / 'faculty_preferences.csv', index=False)
        datasets['room_availability'].to_csv(output_path / 'room_availability.csv', index=False)
        
        # Save metadata
        metadata = {
            'num_courses': self.num_courses,
            'num_students': self.num_students,
            'num_rooms': self.num_rooms,
            'num_faculty': self.num_faculty,
            'num_days': self.num_days,
            'num_timeslots': self.num_timeslots,
            'total_timeslots': self.total_timeslots,
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ All files saved to {output_dir}/")
        
        return output_path
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("EXAM SCHEDULING DATASET SUMMARY")
        print("="*60)
        
        print(f"\nProblem Size:")
        print(f"  Courses:              {self.num_courses}")
        print(f"  Students:             {self.num_students}")
        print(f"  Rooms:                {self.num_rooms}")
        print(f"  Faculty:              {self.num_faculty}")
        print(f"  Exam Days:            {self.num_days}")
        print(f"  Timeslots/Day:        {self.num_timeslots}")
        print(f"  Total Timeslots:      {self.total_timeslots}")
        
        if self.conflict_matrix is not None:
            conflicts = self.conflict_matrix[self.conflict_matrix > 0]
            print(f"\nConflict Statistics:")
            print(f"  Total Conflict Pairs: {len(conflicts)}")
            print(f"  Max Conflict Size:    {self.conflict_matrix.max()}")
        
        print("\n" + "="*60)


# Generate multiple dataset sizes
def generate_multiple_sizes():
    """Generate datasets of different sizes for testing"""
    
    configs = [
        {'size': 'TINY', 'courses': 4, 'students': 30, 'rooms': 3, 'faculty': 4, 'days': 2, 'slots': 2},
        {'size': 'SMALL', 'courses': 10, 'students': 100, 'rooms': 5, 'faculty': 8, 'days': 3, 'slots': 2},
        {'size': 'MEDIUM', 'courses': 20, 'students': 250, 'rooms': 10, 'faculty': 15, 'days': 5, 'slots': 2},
        {'size': 'LARGE', 'courses': 50, 'students': 500, 'rooms': 20, 'faculty': 30, 'days': 10, 'slots': 2},
    ]
    
    for config in configs:
        print(f"\n{'#'*60}")
        print(f"# Generating {config['size']} dataset")
        print(f"{'#'*60}")
        
        generator = ExamSchedulingDataset(
            num_courses=config['courses'],
            num_students=config['students'],
            num_rooms=config['rooms'],
            num_faculty=config['faculty'],
            num_days=config['days'],
            num_timeslots=config['slots']
        )
        
        generator.generate_all()
        generator.print_summary()
        output_dir = f'./exam_scheduling_data_{config["size"].lower()}'
        generator.save_to_csv(output_dir)


if __name__ == '__main__':
    # Generate multiple dataset sizes
    generate_multiple_sizes()
