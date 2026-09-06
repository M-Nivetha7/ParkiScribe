#knee tracker
import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Calculate angle between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# Start webcam
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    count = 0
    stage = None
    start_time = time.time()
    knee_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 30:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)
            knee_angles.append(angle)

            if angle > 160:
                stage = "up"
            if angle < 90 and stage == "up":
                stage = "down"
                count += 1

            # Display info
            cv2.putText(image, f'Time: {int(elapsed_time)}s', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(image, f'REPS: {count}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(image, f'Angle: {int(angle)}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Knee Exercise Tracker (30s)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    avg_angle = int(np.mean(knee_angles)) if knee_angles else 0
    reps_per_sec = round(count / 30, 2)
    feedback = "Nice form!" if count >= 10 else "Try deeper knee bends."

    print("\n--- KNEE EXERCISE SUMMARY ---")
    print(f"Duration: 30 seconds")
    print(f"Total Repetitions: {count}")
    print(f"Average Knee Angle: {avg_angle} degrees")
    print(f"Reps Per Second: {reps_per_sec}")
    print(f"Feedback: {feedback}")

#elbow tracker
import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    return 360 - angle if angle > 180 else angle

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    count = 0
    stage = None
    start_time = time.time()
    elbow_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > 30:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)
            elbow_angles.append(angle)

            if angle > 160:
                stage = "down"
            if angle < 40 and stage == "down":
                stage = "up"
                count += 1

            cv2.putText(image, f'Time: {int(elapsed_time)}s', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(image, f'REPS: {count}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(image, f'Angle: {int(angle)}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Elbow Tracker (30s)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    avg_angle = int(np.mean(elbow_angles)) if elbow_angles else 0
    reps_per_sec = round(count / 30, 2)
    feedback = "Great elbow control!" if count >= 10 else "Try to keep consistent form."

    print("\n--- ELBOW EXERCISE SUMMARY ---")
    print(f"Duration: 30 seconds")
    print(f"Total Repetitions: {count}")
    print(f"Average Elbow Angle: {avg_angle} degrees")
    print(f"Reps Per Second: {reps_per_sec}")
    print(f"Feedback: {feedback}")

#session_data = []

# Example entry
#session_data.append({
#   'timestamp': '2025-05-24 15:00',
 #   'exercise': 'shoulder raise',
  #  'angle': 45,
   # 'reps': 1
#})


import cv2
import mediapipe as mp
import pandas as pd
import math
import datetime

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize data storage
session_data = []
reps = 0
last_angle = None

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    angle = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to stop recording and save data.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks = results.pose_landmarks.landmark
        
        # Get shoulder, elbow, wrist
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        
        # Calculate shoulder angle
        angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        
        # Display angle
        cv2.putText(frame, f'Shoulder Angle: {int(angle)}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Simple rep counting (example logic: crossing below 70 degrees = count)
        if last_angle is not None:
            if last_angle > 70 and angle <= 70:
                reps += 1
                print(f"Rep counted! Total reps: {reps}")
        
        last_angle = angle
        
        # Save data point
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_data.append({
            'timestamp': timestamp,
            'exercise': 'shoulder raise',
            'angle': angle,
            'reps': reps
        })
    
    cv2.imshow('Shoulder Exercise Tracker', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# After collecting data
cap.release()
cv2.destroyAllWindows()

# Convert session data (list of dicts) to DataFrame
df = pd.DataFrame(session_data)

# Save to Excel at desired location
df.to_excel('/Users/nivetham/Downloads/exercise_summary.xlsx', index=False)
print("Saved exercise summary to '/Users/nivetham/Downloads/exercise_summary.xlsx'")


import tkinter as tk
from tkinter import filedialog

# Initialize Tkinter
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open file dialog
file_path = filedialog.askopenfilename()
print(f"Selected file: {'/Users/nivetham/Downloads/exercise_summary.xlsx'}")


import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Setup webcam
cap = cv2.VideoCapture(0)

# Data storage
shoulder_data = []
elbow_data = []
knee_data = []

# Angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recolor image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate angles
        shoulder_angle = calculate_angle(hip, shoulder, elbow)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        knee_angle = calculate_angle(hip, knee, ankle)

        # Save timestamp
        timestamp = datetime.now()

        # Store data
        shoulder_data.append({'timestamp': timestamp, 'shoulder_angle': shoulder_angle})
        elbow_data.append({'timestamp': timestamp, 'elbow_angle': elbow_angle})
        knee_data.append({'timestamp': timestamp, 'knee_angle': knee_angle})

        # Display angles on video
        cv2.putText(image, f'Shoulder: {int(shoulder_angle)} deg', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Elbow: {int(elbow_angle)} deg', (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Knee: {int(knee_angle)} deg', (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    except:
        pass

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Live Exercise Tracker', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert to DataFrames
df_shoulder = pd.DataFrame(shoulder_data)
df_elbow = pd.DataFrame(elbow_data)
df_knee = pd.DataFrame(knee_data)

# Save to separate Excel files
df_shoulder.to_excel('shoulder_tracker.xlsx', index=False)
df_elbow.to_excel('Elbow_tracker.xlsx', index=False)
df_knee.to_excel('Knee_tracker.xlsx', index=False)

print('✅ All exercise data saved:')
print('- Shoulder → Shoulder_tracker.xlsx')
print('- Elbow → Elbow_tracker.xlsx')
print('- Knee → Knee_tracker.xlsx')


import pandas as pd

df = pd.read_excel('//Users/nivetham/Documents/AR:XR/shoulder_tracker.xlsx')  # if in the same folder
print(df.head())


# Assuming first column is X, second is Y
X = df.iloc[:, 0].values.reshape(-1, 1)  # input feature as 2D array
y = df.iloc[:, 1].values  # target


from sklearn.linear_model import LinearRegression

# Initialize model
model = LinearRegression()

# Train (fit)
model.fit(X, y)


# Basic stats on the angle column
summary_stats = df['shoulder_angle'].describe()
print(summary_stats)


df.columns = df.columns.str.strip().str.lower()
print(df.columns)


summary_stats = df['shoulder_angle'].describe()


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['shoulder_angle'], label='Shoulder Angle', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Angle (degrees)')
plt.title('Shoulder Angle Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('shoulder_angle_over_time.png')
plt.show()


import sys
print(sys.executable)


import sys
!{sys.executable} -m pip install matplotlib


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['shoulder_angle'], label='Shoulder Angle', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Angle (degrees)')
plt.title('Shoulder Angle Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('shoulder_angle_over_time.png')
plt.show()


# Define threshold
threshold = 45

# Track states
above_threshold = False
reps = 0

for angle in df['shoulder_angle']:
    if angle > threshold and not above_threshold:
        above_threshold = True
    if angle < threshold and above_threshold:
        reps += 1
        above_threshold = False

print(f"Total repetitions detected: {reps}")


# Add a summary row at the end
summary_row = {'timestamp': 'Summary', 'exercise': 'shoulder raise', 'shoulder_angle': df['shoulder_angle'].mean(), 'reps': reps}
df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

# Save to a new Excel file
df.to_excel('shoulder_tracker_processed.xlsx', index=False)

print("Processed Excel saved as shoulder_tracker_processed.xlsx")


summary_stats = df['shoulder_angle'].describe()


summary_stats = df['shoulder_angle'].describe()
print(summary_stats)


threshold = 45
above_threshold = False
reps = 0

for angle in df['shoulder_angle']:
    if angle > threshold and not above_threshold:
        above_threshold = True
    if angle < threshold and above_threshold:
        reps += 1
        above_threshold = False

print(f"Total repetitions detected: {reps}")


summary_row = {
    'timestamp': 'Summary',
    'exercise': 'shoulder raise',
    'shoulder_angle': df['shoulder_angle'].mean(),
    'reps': reps
}
df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

df.to_excel('shoulder_tracker_processed.xlsx', index=False)

print("Processed Excel saved as shoulder_tracker_processed.xlsx")


print(df.columns)


df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')


df = df.dropna(subset=['timestamp'])


df = df.sort_values('timestamp')


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['shoulder_angle'], label='Shoulder Angle', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Angle (degrees)')
plt.title('Shoulder Angle Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def analyze_exercise(file_in, angle_column, file_out):
    # your processing code here

    # Shoulder
    analyze_exercise('shoulder_tracker.xlsx', 'shoulder_angle', 'shoulder_tracker_processed.xlsx')

    # Elbow
    analyze_exercise('elbow_tracker.xlsx', 'elbow_angle', 'elbow_tracker_processed.xlsx')

    # Knee
    analyze_exercise('knee_tracker.xlsx', 'knee_angle', 'knee_tracker_processed.xlsx')


import pandas as pd
import matplotlib.pyplot as plt


# Example: compare average angles
shoulder_df = pd.read_excel('shoulder_tracker.xlsx')
elbow_df = pd.read_excel('elbow_tracker.xlsx')
knee_df = pd.read_excel('knee_tracker.xlsx')

# Calculate averages
avg_angles = {
    'Shoulder': shoulder_df['shoulder_angle'].mean(),
    'Elbow': elbow_df['elbow_angle'].mean(),
    'Knee': knee_df['knee_angle'].mean()
}

# Bar graph
plt.figure(figsize=(8, 6))
plt.bar(avg_angles.keys(), avg_angles.values(), color=['blue', 'green', 'orange'])
plt.xlabel('Exercise')
plt.ylabel('Average Angle (degrees)')
plt.title('Average Angle per Exercise')
plt.show()
plt.savefig("/Users/nivetham/Documents/AR:XR/BAR_CHART")


def count_reps(angle_series, threshold=45):
    if angle_series.empty:
        return 0
    above = False
    reps = 0
    for angle in angle_series.dropna():
        if angle > threshold and not above:
            above = True
        if angle < threshold and above:
            reps += 1
            above = False
    return reps


shoulder_reps = count_reps(shoulder_df.get('shoulder_angle', pd.Series()))
elbow_reps = count_reps(elbow_df.get('elbow_angle', pd.Series()))
knee_reps = count_reps(knee_df.get('knee_angle', pd.Series()))

print("Shoulder reps:", shoulder_reps)
print("Elbow reps:", elbow_reps)
print("Knee reps:", knee_reps)


reps = [shoulder_reps, elbow_reps, knee_reps]
labels = ['Shoulder', 'Elbow', 'Knee']
colors = ['blue', 'green', 'orange']

# Avoid pie chart if all are zero or NaN
if sum(reps) > 0:
    plt.figure(figsize=(8, 8))
    plt.pie(reps, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Repetitions Distribution Across Exercises')
    plt.show()
else:
    print("No valid reps data to plot a pie chart.")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("Exercise Tracker: Shoulder, Elbow, Knee")

# Upload Excel files
shoulder_file = st.file_uploader("Upload Shoulder Excel", type=['xlsx'])
elbow_file = st.file_uploader("Upload Elbow Excel", type=['xlsx'])
knee_file = st.file_uploader("Upload Knee Excel", type=['xlsx'])

def count_reps(angle_series, threshold=45):
    if angle_series.empty:
        return 0
    above = False
    reps = 0
    for angle in angle_series.dropna():
        if angle > threshold and not above:
            above = True
        if angle < threshold and above:
            reps += 1
            above = False
    return reps

if shoulder_file and elbow_file and knee_file:
    shoulder_df = pd.read_excel(shoulder_file)
    elbow_df = pd.read_excel(elbow_file)
    knee_df = pd.read_excel(knee_file)
    
    # Show raw data
    st.subheader("Uploaded Data")
    st.write("Shoulder Data", shoulder_df.head())
    st.write("Elbow Data", elbow_df.head())
    st.write("Knee Data", knee_df.head())

    # Plot line charts
    st.subheader("Angle Over Time")
    fig, ax = plt.subplots()
    ax.plot(shoulder_df['timestamp'], shoulder_df['shoulder_angle'], label='Shoulder', color='blue')
    ax.plot(elbow_df['timestamp'], elbow_df['elbow_angle'], label='Elbow', color='green')
    ax.plot(knee_df['timestamp'], knee_df['knee_angle'], label='Knee', color='orange')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Angle (degrees)')
    ax.legend()
    st.pyplot(fig)

    # Calculate and show reps
    shoulder_reps = count_reps(shoulder_df['shoulder_angle'])
    elbow_reps = count_reps(elbow_df['elbow_angle'])
    knee_reps = count_reps(knee_df['knee_angle'])

    st.subheader("Repetition Counts")
    st.write(f"Shoulder Reps: {shoulder_reps}")
    st.write(f"Elbow Reps: {elbow_reps}")
    st.write(f"Knee Reps: {knee_reps}")

    # Show pie chart
    reps = [shoulder_reps, elbow_reps, knee_reps]
    labels = ['Shoulder', 'Elbow', 'Knee']
    colors = ['blue', 'green', 'orange']
    fig2, ax2 = plt.subplots()
    ax2.pie(reps, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax2.set_title('Repetitions Distribution')
    st.pyplot(fig2)

    # Download report button
    st.subheader("Download Reports")
    with pd.ExcelWriter('exercise_report.xlsx') as writer:
        shoulder_df.to_excel(writer, sheet_name='Shoulder', index=False)
        elbow_df.to_excel(writer, sheet_name='Elbow', index=False)
        knee_df.to_excel(writer, sheet_name='Knee', index=False)
    with open('exercise_report.xlsx', 'rb') as f:
        st.download_button('Download Full Report', f, file_name='exercise_report.xlsx')


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("Exercise Tracker: Shoulder, Elbow, Knee")

# Upload Excel files
shoulder_file = st.file_uploader("Upload Shoulder Excel", type=['xlsx'])
elbow_file = st.file_uploader("Upload Elbow Excel", type=['xlsx'])
knee_file = st.file_uploader("Upload Knee Excel", type=['xlsx'])

def count_reps(angle_series, threshold=45):
    if angle_series.empty:
        return 0
    above = False
    reps = 0
    for angle in angle_series.dropna():
        if angle > threshold and not above:
            above = True
        if angle < threshold and above:
            reps += 1
            above = False
    return reps

def show_statistics(df, angle_column):
    st.write(f"**Mean**: {df[angle_column].mean():.2f}")
    st.write(f"**Median**: {df[angle_column].median():.2f}")
    st.write(f"**Std Deviation**: {df[angle_column].std():.2f}")
    st.write(f"**Min**: {df[angle_column].min():.2f}")
    st.write(f"**Max**: {df[angle_column].max():.2f}")

if shoulder_file and elbow_file and knee_file:
    shoulder_df = pd.read_excel(shoulder_file)
    elbow_df = pd.read_excel(elbow_file)
    knee_df = pd.read_excel(knee_file)
    
    # Show raw data
    st.subheader("Uploaded Data")
    st.write("Shoulder Data", shoulder_df.head())
    st.write("Elbow Data", elbow_df.head())
    st.write("Knee Data", knee_df.head())

    # Plot line charts
    st.subheader("Angle Over Time")
    fig, ax = plt.subplots()
    ax.plot(shoulder_df['timestamp'], shoulder_df['shoulder_angle'], label='Shoulder', color='blue')
    ax.plot(elbow_df['timestamp'], elbow_df['elbow_angle'], label='Elbow', color='green')
    ax.plot(knee_df['timestamp'], knee_df['knee_angle'], label='Knee', color='orange')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Angle (degrees)')
    ax.legend()
    st.pyplot(fig)

    # Calculate and show reps
    shoulder_reps = count_reps(shoulder_df['shoulder_angle'])
    elbow_reps = count_reps(elbow_df['elbow_angle'])
    knee_reps = count_reps(knee_df['knee_angle'])

    st.subheader("Repetition Counts")
    st.write(f"Shoulder Reps: {shoulder_reps}")
    st.write(f"Elbow Reps: {elbow_reps}")
    st.write(f"Knee Reps: {knee_reps}")

    # Show pie chart
    reps = [shoulder_reps, elbow_reps, knee_reps]
    labels = ['Shoulder', 'Elbow', 'Knee']
    colors = ['blue', 'green', 'orange']
    fig2, ax2 = plt.subplots()
    ax2.pie(reps, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax2.set_title('Repetitions Distribution')
    st.pyplot(fig2)

    # Show statistical analysis
    st.subheader("Statistical Analysis")

    st.write("### Shoulder Statistics")
    show_statistics(shoulder_df, 'shoulder_angle')

    st.write("### Elbow Statistics")
    show_statistics(elbow_df, 'elbow_angle')

    st.write("### Knee Statistics")
    show_statistics(knee_df, 'knee_angle')

    # Download report button
    st.subheader("Download Reports")
    with pd.ExcelWriter('exercise_report.xlsx') as writer:
        shoulder_df.to_excel(writer, sheet_name='Shoulder', index=False)
        elbow_df.to_excel(writer, sheet_name='Elbow', index=False)
        knee_df.to_excel(writer, sheet_name='Knee', index=False)
    with open('exercise_report.xlsx', 'rb') as f:
        st.download_button('Download Full Report', f, file_name='exercise_report.xlsx')


import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
shoulder_df = pd.read_excel('shoulder_tracker.xlsx')
elbow_df = pd.read_excel('elbow_tracker.xlsx')
knee_df = pd.read_excel('knee_tracker.xlsx')

# Function to generate graphs for each dataset
def analyze_and_plot(df, joint_name):
    # Convert 'timestamp' to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    angle_col = f'{joint_name}_angle'

    # Line plot: angle over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df[angle_col], label=f'{joint_name.capitalize()} Angle', color='blue')
    plt.xlabel('Timestamp')
    plt.ylabel('Angle (degrees)')
    plt.title(f'{joint_name.capitalize()} Angle Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Bar chart: average, min, max
    avg_angle = df[angle_col].mean()
    min_angle = df[angle_col].min()
    max_angle = df[angle_col].max()

    plt.figure(figsize=(6, 6))
    plt.bar(['Average', 'Minimum', 'Maximum'], [avg_angle, min_angle, max_angle], color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('Angle (degrees)')
    plt.title(f'{joint_name.capitalize()} Angle Statistics')
    plt.show()

    # Pie chart: share of repetitions (assuming one rep per row, else adjust logic)
    reps = [len(df)]
    other_reps = 100 - len(df) if 100 - len(df) > 0 else 0
    labels = [f'{joint_name.capitalize()} Reps', 'Other']
    sizes = [reps[0], other_reps]
    colors = ['gold', 'lightgray']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f'{joint_name.capitalize()} Repetition Share')
    plt.show()

    # Summary stats printout
    print(f"\n=== {joint_name.capitalize()} Summary Statistics ===")
    print(df[angle_col].describe())
    print("\n")

# Run for all three joints
analyze_and_plot(shoulder_df, 'shoulder')
analyze_and_plot(elbow_df, 'elbow')
analyze_and_plot(knee_df, 'knee')


import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd



# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

important_body_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
]


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle



class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.left_angle = 0
        self.right_angle = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mask = img.copy()
            landmarks = results.pose_landmarks.landmark

            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                x1, y1 = int(start.x * img.shape[1]), int(start.y * img.shape[0])
                x2, y2 = int(end.x * img.shape[1]), int(end.y * img.shape[0])
                cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 255), 2)

            for idx, l in enumerate(landmarks):
                x, y = int(l.x * img.shape[1]), int(l.y * img.shape[0])
                if idx in important_body_indices:
                    cv2.circle(mask, (x, y), 8, (0, 255, 0), -1)
                    cv2.circle(mask, (x, y), 12, (0, 180, 0), 2)
                else:
                    cv2.circle(mask, (x, y), 3, (0, 0, 255), -1)
                    cv2.circle(mask, (x, y), 6, (0, 0, 180), 1)

            # Calculate angles
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1],
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0]]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * img.shape[1],
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * img.shape[0]]
            self.left_angle = 180 - calculate_angle(left_elbow, left_shoulder,
                                                    [left_shoulder[0], left_shoulder[1] - 100])

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1],
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0]]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * img.shape[1],
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * img.shape[0]]
            self.right_angle = 180 - calculate_angle(right_elbow, right_shoulder,
                                                     [right_shoulder[0], right_shoulder[1] - 100])

            return mask
        return img



def draw_angle_meter(angle, label):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    circle = plt.Circle((0, 0), 1, color=(0.2, 0.2, 0.2), fill=True)
    ax.add_artist(circle)

    if angle > 120:
        arc_color = "#4CAF50"  # Green
        level = "Excellent"
    elif 60 < angle <= 120:
        arc_color = "#FFC107"  # Amber
        level = "Moderate"
    else:
        arc_color = "#F44336"  # Red
        level = "Needs Work"

    theta = np.linspace(np.pi, np.pi - (np.pi * (angle / 180)), 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color=arc_color, linewidth=8)

    ax.text(0, 0, f"{int(angle)}°", ha='center', va='center', fontsize=20, color='white', fontweight='bold')
    ax.text(0, -1.3, label, ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    ax.text(0, 1.2, level, ha='center', va='center', fontsize=14, color=arc_color, fontweight='bold')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', transparent=True, dpi=120)
    buf.seek(0)
    return buf

def main():
    # Custom CSS for styling
    st.set_page_config(
        layout="wide",
        page_title="NeuroTrack Pro | Stroke Therapy Monitoring",
        page_icon="🧠"
    )
    st.markdown(
        <style>
            .main {
                background-color: #f5f9fc;
            }
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4f0fb 100%);
            }
            .header {
                color: #2c3e50;
                padding: 1rem;
                border-radius: 10px;
                background: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .card {
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1.5rem;
            }
            .metric-card {
                background: white;
                border-radius: 10px;
                padding: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            .timer {
                font-size: 1.2rem;
                color: #3498db;
                font-weight: bold;
            }
            .download-btn {
                background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
                color: white !important;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: bold;
            }
            .stButton>button {
                background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: bold;
            }
            .success-box {
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1.5rem;
            }
        </style>
        , unsafe_allow_html=True
    )


    # Header Section
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; color:#2c3e50;">🧠 NeuroTrack Pro</h1>
        <p style="margin:0; color:#7f8c8d;">AI-Powered Stroke Rehabilitation Progress Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    # Session states
    if "left_angles" not in st.session_state:
        st.session_state.left_angles = []
    if "right_angles" not in st.session_state:
        st.session_state.right_angles = []
    if "timestamps" not in st.session_state:
        st.session_state.timestamps = []
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "last_df" not in st.session_state:
        st.session_state.last_df = None
    if "last_left_meter" not in st.session_state:
        st.session_state.last_left_meter = None
    if "last_right_meter" not in st.session_state:
        st.session_state.last_right_meter = None

    # Main columns layout
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color:#2c3e50; margin-bottom:1rem;">Live Motion Analysis</h3>
        """, unsafe_allow_html=True)

        ctx = webrtc_streamer(
            key="stream",
            video_transformer_factory=VideoTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#2c3e50; margin-bottom:1rem;">Left Arm Mobility</h4>
        """, unsafe_allow_html=True)
        meter_left = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#2c3e50; margin-bottom:1rem;">Right Arm Mobility</h4>
        """, unsafe_allow_html=True)
        meter_right = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom section
    st.markdown("""
    <div class="card">
        <h3 style="color:#2c3e50; margin-bottom:1rem;">Progress Over Time</h3>
    """, unsafe_allow_html=True)
    graph_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # Timer section
    st.markdown("""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="color:#2c3e50; margin:0;">Session Details</h3>
            <div class="timer" id="timer">00:00</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    timer_placeholder = st.empty()

    # Main processing loop
    while ctx.state.playing:
        if ctx.video_transformer:
            left_angle = ctx.video_transformer.left_angle
            right_angle = ctx.video_transformer.right_angle

            st.session_state.left_angles.append(left_angle)
            st.session_state.right_angles.append(right_angle)
            st.session_state.timestamps.append(time.time())

            buf_left = draw_angle_meter(left_angle, "Left Arm")
            buf_right = draw_angle_meter(right_angle, "Right Arm")
            meter_left.image(buf_left)
            meter_right.image(buf_right)

            st.session_state.last_left_meter = buf_left
            st.session_state.last_right_meter = buf_right

            df = pd.DataFrame({
                "Time": st.session_state.timestamps,
                "Left Arm Angle": st.session_state.left_angles,
                "Right Arm Angle": st.session_state.right_angles
            })
            df["Relative Time"] = df["Time"] - df["Time"].iloc[0]
            graph_placeholder.line_chart(df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])

            st.session_state.last_df = df

            elapsed = int(time.time() - st.session_state.start_time)
            mins, secs = divmod(elapsed, 60)
            timer_placeholder.markdown(f"""
            <div class="timer">
                ⏳ Session Duration: {mins:02d}:{secs:02d}
            </div>
            """, unsafe_allow_html=True)

        time.sleep(0.1)

    # After stop
    if not ctx.state.playing:
        if st.session_state.last_df is not None:
            st.markdown("""
            <div class="success-box">
                <h3 style="color:white; margin:0;">✅ Session Completed Successfully!</h3>
                <p style="color:white; margin:0;">Your rehabilitation data has been recorded.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.last_left_meter and st.session_state.last_right_meter:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="color:#2c3e50; margin-bottom:1rem;">Final Left Arm Reading</h4>
                    """, unsafe_allow_html=True)
                    st.image(st.session_state.last_left_meter)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="color:#2c3e50; margin-bottom:1rem;">Final Right Arm Reading</h4>
                    """, unsafe_allow_html=True)
                    st.image(st.session_state.last_right_meter)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <h3 style="color:#2c3e50; margin-bottom:1rem;">Session Summary</h3>
            """, unsafe_allow_html=True)
            st.line_chart(st.session_state.last_df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])
            st.markdown("</div>", unsafe_allow_html=True)

            csv = st.session_state.last_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Session Data (CSV)",
                data=csv,
                file_name="neurotrack_session_data.csv",
                mime="text/csv",
                key="download-csv"
            )

def main():
    # Custom CSS for styling
    st.set_page_config(
        layout="wide",
        page_title="NeuroTrack Pro | Stroke Therapy Monitoring",
        page_icon="🧠"
    )

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import time
import pandas as pd

# Assuming VideoTransformer and draw_angle_meter are already defined

def main():
    # Page config
    st.set_page_config(
        layout="wide",
        page_title="NeuroTrack Pro | Stroke Therapy Monitoring",
        page_icon="🧠"
    )

    # CSS styling
    st.markdown("""
    <style>
        .main {
            background-color: #f5f9fc;
        }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4f0fb 100%);
        }
        .header {
            color: #2c3e50;
            padding: 1rem;
            border-radius: 10px;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .timer {
            font-size: 1.2rem;
            color: #3498db;
            font-weight: bold;
        }
        .download-btn {
            background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stButton>button {
            background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .success-box {
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; color:#2c3e50;">🧠 NeuroTrack Pro</h1>
        <p style="margin:0; color:#7f8c8d;">AI-Powered Stroke Rehabilitation Progress Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    # Session states
    for key in ["left_angles", "right_angles", "timestamps"]:
        if key not in st.session_state:
            st.session_state[key] = []

    st.session_state.setdefault("start_time", time.time())
    st.session_state.setdefault("last_df", None)
    st.session_state.setdefault("last_left_meter", None)
    st.session_state.setdefault("last_right_meter", None)

    # Layout columns
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color:#2c3e50; margin-bottom:1rem;">Live Motion Analysis</h3>
        """, unsafe_allow_html=True)

        ctx = webrtc_streamer(
            key="stream",
            video_transformer_factory=VideoTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#2c3e50; margin-bottom:1rem;">Left Arm Mobility</h4>
        """, unsafe_allow_html=True)
        meter_left = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#2c3e50; margin-bottom:1rem;">Right Arm Mobility</h4>
        """, unsafe_allow_html=True)
        meter_right = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    # Progress Graph
    st.markdown("""
    <div class="card">
        <h3 style="color:#2c3e50; margin-bottom:1rem;">Progress Over Time</h3>
    """, unsafe_allow_html=True)
    graph_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # Timer
    st.markdown("""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="color:#2c3e50; margin:0;">Session Details</h3>
            <div class="timer" id="timer">00:00</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    timer_placeholder = st.empty()

    # Main loop
    while ctx.state.playing:
        if ctx.video_transformer:
            left = ctx.video_transformer.left_angle
            right = ctx.video_transformer.right_angle

            st.session_state.left_angles.append(left)
            st.session_state.right_angles.append(right)
            st.session_state.timestamps.append(time.time())

            buf_left = draw_angle_meter(left, "Left Arm")
            buf_right = draw_angle_meter(right, "Right Arm")
            meter_left.image(buf_left)
            meter_right.image(buf_right)

            st.session_state.last_left_meter = buf_left
            st.session_state.last_right_meter = buf_right

            df = pd.DataFrame({
                "Time": st.session_state.timestamps,
                "Left Arm Angle": st.session_state.left_angles,
                "Right Arm Angle": st.session_state.right_angles
            })
            df["Relative Time"] = df["Time"] - df["Time"].iloc[0]
            graph_placeholder.line_chart(df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])

            st.session_state.last_df = df

            elapsed = int(time.time() - st.session_state.start_time)
            mins, secs = divmod(elapsed, 60)
            timer_placeholder.markdown(f"""
            <div class="timer">
                ⏳ Session Duration: {mins:02d}:{secs:02d}
            </div>
            """, unsafe_allow_html=True)

        time.sleep(0.1)

    # After session ends
    if not ctx.state.playing and st.session_state.last_df is not None:
        st.markdown("""
        <div class="success-box">
            <h3 style="color:white; margin:0;">✅ Session Completed Successfully!</h3>
            <p style="color:white; margin:0;">Your rehabilitation data has been recorded.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.last_left_meter and st.session_state.last_right_meter:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color:#2c3e50; margin-bottom:1rem;">Final Left Arm Reading</h4>
                """, unsafe_allow_html=True)
                st.image(st.session_state.last_left_meter)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color:#2c3e50; margin-bottom:1rem;">Final Right Arm Reading</h4>
                """, unsafe_allow_html=True)
                st.image(st.session_state.last_right_meter)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3 style="color:#2c3e50; margin-bottom:1rem;">Session Summary</h3>
        """, unsafe_allow_html=True)
        st.line_chart(st.session_state.last_df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])
        st.markdown("</div>", unsafe_allow_html=True)

        csv = st.session_state.last_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Session Data (CSV)",
            data=csv,
            file_name="neurotrack_session_data.csv",
            mime="text/csv",
            key="download-csv"
        )
if __name__ == "__main__": 
    main()


import streamlit as st
from streamlit_webrtc import webrtc_streamer
import time
import pandas as pd


if __name__ == "__main__":
    main()



