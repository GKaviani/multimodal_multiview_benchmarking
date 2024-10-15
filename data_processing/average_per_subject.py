import os
import csv
import numpy as np
from collections import defaultdict

def compute_activity_duration(base_dir):
    # Dictionary to store total duration and session counts per subject
    subject_data = defaultdict(lambda: defaultdict(list))
    # Traverse the directory structure
    for activity_class in os.listdir(base_dir):
        activity_path = os.path.join(base_dir, activity_class, 'RGB', 'camera_1_fps_15')
        if os.path.isdir(activity_path):
            for subject_id in os.listdir(activity_path):
                subject_path = os.path.join(activity_path, subject_id)
                if os.path.isdir(subject_path):
                    for session_id in os.listdir(subject_path):
                        session_path = os.path.join(subject_path, session_id)
                        if os.path.isdir(session_path):
                            print(f"we are computing {session_path}")
                            # Count the number of PNG files
                            png_count = len([f for f in os.listdir(session_path) if f.endswith('.png')])
                            # Compute duration for this session
                            duration = png_count / 15.0
                            subject_data[subject_id][activity_class].append({
                                'duration': duration,
                                'session_id': session_id
                            })
    print("Done!")

    return subject_data

def compute_average_activity_duration(subject_duration):
    # Dictionary to store average duration per subject and activity
    average_durations = defaultdict(dict)

    for subject_id, activities in subject_duration.items():
        for activity_name, sessions in activities.items():
            total_duration = sum(session['duration'] for session in sessions)
            session_count = len(sessions)
            average_duration = total_duration / session_count if session_count > 0 else 0
            average_durations[subject_id][activity_name] = average_duration

    return average_durations

def save_to_csv(average_durations, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['subject_id', 'activity_name', 'average_duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for subject_id, activities in average_durations.items():
            for activity_name, avg_duration in activities.items():
                writer.writerow({
                    'subject_id': subject_id,
                    'activity_name': activity_name,
                    'average_duration': avg_duration
                })



# Example usage
# base_dir = '/mnt/data-tmp/ghazal/DARai_DATA/L1_Activities/'
# output_file = './subject_activity_average_durations.csv'
# subject_data = compute_activity_duration(base_dir)
# average_activity_durations = compute_average_activity_duration(subject_data)
# save_to_csv(average_activity_durations, output_file)
#
# print(f"Average durations saved to {output_file}")
# print("Subject Data:", subject_data)
# print("Average Activity Durations:", average_activity_durations)

import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data
data = pd.read_csv('/home/ghazal/Activity_Recognition_benchmarking/subject_activity_average_durations.csv')

# Create a grouped bar chart
subjects = sorted(data['subject_id'].unique())
activities = sorted(data['activity_name'].unique())

# Set the width for each bar
bar_width = 0.15
spacing = 0.05
index = np.arange(len(subjects))

# Plot the bars
plt.figure(figsize=(20, 12))
for i, activity in enumerate(activities):
    # Filter the data for the current activity
    activity_data = data[data['activity_name'] == activity]
    # Make sure the data is aligned with the subjects order
    activity_data = activity_data.set_index('subject_id').reindex(subjects)
    plt.bar(index + i * (bar_width + spacing), activity_data['average_duration'],
            width=bar_width, label=activity)

plt.xlabel('Subject ID', fontsize=18, fontweight='bold')
plt.ylabel('Average Duration (seconds)', fontsize=18, fontweight='bold')
plt.title('Average Activity Duration per Subject', fontsize=22, fontweight='bold')
plt.xticks(index + (bar_width + spacing) * (len(activities) / 2), subjects,
           rotation=45, fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

plt.legend(title='Activity', fontsize=14, title_fontsize=16)

plt.tight_layout()
plt.savefig("./subject_activity_average_durations.png", format='png')
plt.savefig("./subject_activity_average_durations.svg", format='svg')
plt.show()