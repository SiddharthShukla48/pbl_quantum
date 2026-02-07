"""
Graph Coloring Exam Scheduling - Dataset Generator
===================================================

This file generates realistic exam scheduling data for 2nd and 3rd year students.

What it creates:
1. courses.csv - List of courses with enrollment numbers
2. students.csv - Student population
3. enrollments.csv - Which students take which courses (THIS IS KEY!)
4. rooms.csv - Available rooms with capacities
5. conflict_graph.csv - Which exams conflict (derived from enrollments)

Author: For quantum graph coloring exam scheduling research
Date: February 2026
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path
import json
from datetime import datetime


class ExamSchedulingDataset:
    """
    Generate exam scheduling datasets for graph coloring approach
    
    Key concept: The conflict graph is built from ACTUAL student enrollments.
    If two courses share students, they MUST be in different time slots.
    """
    
    def __init__(self, 
                 num_courses=10,
                 num_students=50,
                 num_rooms=5,
                 avg_courses_per_student=4,
                 random_seed=42):
        """
        Initialize dataset generator
        
        Args:
            num_courses: Number of courses/exams to schedule
            num_students: Total number of 2nd/3rd year students
            num_rooms: Number of available rooms
            avg_courses_per_student: How many courses each student typically takes
            random_seed: For reproducibility
        """
        self.num_courses = num_courses
        self.num_students = num_students
        self.num_rooms = num_rooms
        self.avg_courses_per_student = avg_courses_per_student
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Storage for generated data
        self.courses = None
        self.students = None
        self.enrollments = None
        self.rooms = None
        self.conflict_adjacency = None  # For graph coloring
        self.conflict_counts = None     # How many students in both courses
        
    def generate_courses(self):
        """
        Generate course list with realistic enrollments
        
        Returns:
            DataFrame with columns: course_id, course_code, year, enrollment
        """
        courses = []
        
        # Predefined course codes for realism
        course_codes_2nd = ['CS201', 'CS202', 'CS203', 'MATH201', 'PHYS201', 
                           'EE201', 'CHEM201', 'BIO201']
        course_codes_3rd = ['CS301', 'CS302', 'CS303', 'MATH301', 'PHYS301',
                           'EE301', 'STAT301', 'AI301']
        
        all_codes = course_codes_2nd + course_codes_3rd
        
        for i in range(self.num_courses):
            # Alternate between 2nd and 3rd year
            year = 2 if i % 2 == 0 else 3
            code = all_codes[i] if i < len(all_codes) else f'COURSE{i:02d}'
            
            # Enrollment: some courses are popular (60-80), some small (20-40)
            if i % 4 == 0:
                enrollment = np.random.randint(60, 80)  # Popular course
            else:
                enrollment = np.random.randint(20, 40)  # Regular course
            
            courses.append({
                'course_id': i,
                'course_code': code,
                'year': year,
                'enrollment': enrollment,
                'duration_hours': np.random.choice([2, 3])  # 2 or 3 hour exam
            })
        
        self.courses = pd.DataFrame(courses)
        print(f"✓ Generated {len(self.courses)} courses")
        return self.courses
    
    def generate_students(self):
        """Generate student population"""
        students = []
        
        for i in range(self.num_students):
            students.append({
                'student_id': i,
                'student_name': f'Student_{i:03d}',
                'year': 2 if i < self.num_students // 2 else 3  # Half in each year
            })
        
        self.students = pd.DataFrame(students)
        print(f"✓ Generated {len(self.students)} students")
        return self.students
    
    def generate_enrollments(self):
        """
        Generate enrollments - THIS IS THE MOST IMPORTANT FUNCTION!
        
        This determines which courses CONFLICT with each other.
        If Student A takes Course 1 AND Course 3, then Courses 1 and 3
        CANNOT be scheduled at the same time.
        
        Returns:
            DataFrame with columns: student_id, course_id, course_code
        """
        enrollments = []
        
        # For each student, randomly assign courses
        for student_id in range(self.num_students):
            student_year = self.students.loc[student_id, 'year']
            
            # Filter courses for this year
            available_courses = self.courses[
                self.courses['year'] == student_year
            ]['course_id'].tolist()
            
            if not available_courses:
                continue
            
            # How many courses does this student take?
            # Normal distribution around avg_courses_per_student
            num_courses_for_student = int(np.random.normal(
                self.avg_courses_per_student, 
                1.0
            ))
            num_courses_for_student = max(2, min(num_courses_for_student, 
                                                 len(available_courses)))
            
            # Randomly select courses
            selected_courses = np.random.choice(
                available_courses,
                size=num_courses_for_student,
                replace=False
            )
            
            # Add enrollments
            for course_id in selected_courses:
                enrollments.append({
                    'student_id': student_id,
                    'course_id': course_id,
                    'course_code': self.courses.loc[course_id, 'course_code']
                })
        
        self.enrollments = pd.DataFrame(enrollments)
        print(f"✓ Generated {len(self.enrollments)} enrollments")
        return self.enrollments
    
    def build_conflict_graph(self):
        """
        Build conflict graph from enrollments
        
        KEY CONCEPT FOR GRAPH COLORING:
        - Vertices = Courses (exams)
        - Edge between i and j if ANY student takes both courses
        - Graph coloring = assigning time slots such that adjacent exams
          get different colors (slots)
        
        Returns:
            tuple: (adjacency_matrix, conflict_counts)
            - adjacency_matrix[i,j] = 1 if courses i,j conflict
            - conflict_counts[i,j] = number of students in both courses
        """
        n = self.num_courses
        adjacency = np.zeros((n, n), dtype=int)
        counts = np.zeros((n, n), dtype=int)
        
        # For each student, find all pairs of their courses
        for student_id in self.enrollments['student_id'].unique():
            # Get all courses this student is enrolled in
            student_courses = self.enrollments[
                self.enrollments['student_id'] == student_id
            ]['course_id'].tolist()
            
            # All pairs of courses this student takes are conflicts
            for i in range(len(student_courses)):
                for j in range(i+1, len(student_courses)):
                    ci, cj = student_courses[i], student_courses[j]
                    
                    # Mark as conflicting
                    adjacency[ci, cj] = 1
                    adjacency[cj, ci] = 1
                    
                    # Count how many students cause this conflict
                    counts[ci, cj] += 1
                    counts[cj, ci] += 1
        
        self.conflict_adjacency = adjacency
        self.conflict_counts = counts
        
        # Count total edges
        num_edges = np.sum(adjacency) // 2  # Divide by 2 since symmetric
        print(f"✓ Built conflict graph: {n} nodes, {num_edges} edges")
        
        return adjacency, counts
    
    def generate_rooms(self):
        """Generate room data with capacities"""
        rooms = []
        
        # Mix of small (30) and large (90) rooms
        for i in range(self.num_rooms):
            if i < self.num_rooms // 3:
                capacity = 90  # Large lecture halls
                room_type = 'Lecture_Hall'
            else:
                capacity = 30  # Regular classrooms
                room_type = 'Classroom'
            
            rooms.append({
                'room_id': i,
                'room_name': f'Room_{i:02d}',
                'capacity': capacity,
                'room_type': room_type,
                'building': 'Main' if i < 3 else 'CS_Block'
            })
        
        self.rooms = pd.DataFrame(rooms)
        print(f"✓ Generated {len(self.rooms)} rooms")
        return self.rooms
    
    def save_to_csv(self, output_dir='./exam_data_graph_coloring'):
        """
        Save all generated data to CSV files
        
        Args:
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate all data if not already done
        if self.courses is None:
            self.generate_courses()
        if self.students is None:
            self.generate_students()
        if self.enrollments is None:
            self.generate_enrollments()
        if self.conflict_adjacency is None:
            self.build_conflict_graph()
        if self.rooms is None:
            self.generate_rooms()
        
        # Save DataFrames to CSV
        self.courses.to_csv(output_path / 'courses.csv', index=False)
        self.students.to_csv(output_path / 'students.csv', index=False)
        self.enrollments.to_csv(output_path / 'enrollments.csv', index=False)
        self.rooms.to_csv(output_path / 'rooms.csv', index=False)
        
        # Save conflict graph as CSV
        conflict_df = pd.DataFrame(
            self.conflict_adjacency,
            index=[f'Course_{i}' for i in range(self.num_courses)],
            columns=[f'Course_{i}' for i in range(self.num_courses)]
        )
        conflict_df.to_csv(output_path / 'conflict_adjacency.csv')
        
        # Save conflict counts
        counts_df = pd.DataFrame(
            self.conflict_counts,
            index=[f'Course_{i}' for i in range(self.num_courses)],
            columns=[f'Course_{i}' for i in range(self.num_courses)]
        )
        counts_df.to_csv(output_path / 'conflict_counts.csv')
        
        # Save metadata
        metadata = {
            'num_courses': self.num_courses,
            'num_students': self.num_students,
            'num_rooms': self.num_rooms,
            'num_enrollments': len(self.enrollments),
            'num_conflicts': int(np.sum(self.conflict_adjacency) // 2),
            'avg_courses_per_student': self.avg_courses_per_student
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ All files saved to {output_dir}/")
        print(f"  - courses.csv ({len(self.courses)} courses)")
        print(f"  - students.csv ({len(self.students)} students)")
        print(f"  - enrollments.csv ({len(self.enrollments)} enrollments)")
        print(f"  - rooms.csv ({len(self.rooms)} rooms)")
        print(f"  - conflict_adjacency.csv (graph structure)")
        print(f"  - conflict_counts.csv (edge weights)")
        
    def print_summary(self):
        """Print summary statistics"""
        if self.conflict_adjacency is None:
            self.build_conflict_graph()
        
        print("\n" + "="*60)
        print("EXAM SCHEDULING DATASET SUMMARY (Graph Coloring)")
        print("="*60)
        
        print(f"\nCourses:  {self.num_courses}")
        print(f"Students: {self.num_students}")
        print(f"Rooms:    {self.num_rooms}")
        print(f"Total Enrollments: {len(self.enrollments)}")
        
        print(f"\nConflict Graph:")
        num_edges = int(np.sum(self.conflict_adjacency) // 2)
        density = num_edges / (self.num_courses * (self.num_courses - 1) / 2) * 100
        print(f"  Edges (conflicts): {num_edges}")
        print(f"  Graph density:     {density:.1f}%")
        
        # Estimate chromatic number (lower bound = max degree + 1)
        degrees = np.sum(self.conflict_adjacency, axis=1)
        max_degree = int(np.max(degrees))
        print(f"  Max degree:        {max_degree}")
        print(f"  Min colors needed: ≥{max_degree + 1} (graph theory bound)")
        
        print("\n" + "="*60)


# Generate multiple dataset sizes
def generate_all_sizes(output_base_dir='./output'):
    """Generate TINY, SMALL, MEDIUM datasets for testing"""
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f'run_{timestamp}'
    datasets_dir = output_dir / 'datasets'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📁 Datasets will be saved to: {datasets_dir}\n")
    
    configs = [
        {'size': 'TINY', 'courses': 5, 'students': 20, 'rooms': 3, 'avg_courses': 3},
        {'size': 'SMALL', 'courses': 10, 'students': 50, 'rooms': 5, 'avg_courses': 4},
        {'size': 'MEDIUM', 'courses': 15, 'students': 80, 'rooms': 7, 'avg_courses': 4},
    ]
    
    for config in configs:
        print(f"\n{'#'*60}")
        print(f"# Generating {config['size']} dataset")
        print(f"{'#'*60}")
        
        generator = ExamSchedulingDataset(
            num_courses=config['courses'],
            num_students=config['students'],
            num_rooms=config['rooms'],
            avg_courses_per_student=config['avg_courses']
        )
        
        generator.generate_courses()
        generator.generate_students()
        generator.generate_enrollments()
        generator.build_conflict_graph()
        generator.generate_rooms()
        
        generator.print_summary()
        
        dataset_output = datasets_dir / f'exam_data_{config["size"].lower()}'
        generator.save_to_csv(dataset_output)
    
    # Save output directory path for next steps
    with open(Path(output_base_dir) / 'latest_run.txt', 'w') as f:
        f.write(str(output_dir))
    
    print(f"\n✅ All datasets saved to: {datasets_dir}")
    print(f"✅ Run directory: {output_dir}")
    
    return output_dir


if __name__ == '__main__':
    print("="*60)
    print("EXAM SCHEDULING DATASET GENERATOR")
    print("For Quantum Graph Coloring Approach")
    print("="*60)
    
    # Generate all three sizes
    output_dir = generate_all_sizes()
    
    print("\n" + "="*60)
    print("DONE! Next steps:")
    print("1. Check the generated directories")
    print("2. Run: python 02_visualize_graph.py")
    print("="*60)
