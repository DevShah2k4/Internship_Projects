from flask import Flask, render_template, request
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import cv2 
app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "database.db")

# ✅ YOUR FOLDERS
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# ===============================
# 🗄️ Create Database
# ===============================
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    #table creation
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crowd_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        result_path TEXT,
        count INTEGER,
        density TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


init_db()
def init_db_video():
    # Connect to database (it will create file if not exists)
    conn = sqlite3.connect("video_data.db")
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT NOT NULL,
        frame_count INTEGER NOT NULL,
        timestamp TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()
init_db_video()
# Load YOLO model
model = YOLO("best.pt")

# ===============================
# Insert Data
# ===============================
def insert_data(image_path, result_path, count, density):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO crowd_data (image_path, result_path, count, density)
    VALUES (?, ?, ?, ?)
    """, (image_path, result_path, count, density))

    conn.commit()
    conn.close()



def save_video_data(video_name, frame_count):
    conn = sqlite3.connect("video_records.db")
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO video_records (video_name, frame_count, timestamp)
    VALUES (?, ?, ?)
    """, (video_name, frame_count, timestamp))

    conn.commit()
    conn.close()

    print("Data saved successfully!")
# =========================
# Video Process Function
# =========================
def process_video(path, filename):
    cap = cv2.VideoCapture(path)

    frame_count = 0
    max_count = 0   # ⭐ track highest crowd

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        count = int(gray.mean() / 10)  

        # The Frame Wise Max Cout it will show 
        if count > max_count:
            max_count = count

    cap.release()

    # ✅ SAVE RESULT FILE
    result_path = os.path.join("static/results", f"{filename}_result.txt")

    with open(result_path, "w") as f:
        f.write(f"Frames: {frame_count}\n")
        f.write(f"Peak Crowd From Frames :{max_count}\n")

    return frame_count, max_count, result_path

    
# =========================
# VIDEO ROUTE
# =========================
@app.route("/video", methods=["GET", "POST"])
def video_page():
    result = None

    if request.method == "POST":
        video = request.files["video"]

        if video.filename == "":
            return "No file selected"

        # ✅ SAVE VIDEO IN static/uploads
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
        video.save(video_path)

        # PROCESS VIDEO
        frames, count, result_file = process_video(video_path, video.filename)

        result = {
            "frames": frames,
            "count": count,
            "file": result_file
        }

    return render_template("video.html", result=result)

# =========================
# LANDING PAGE
# =========================
@app.route("/")
def landing():
    return render_template("landing.html")


# ===============================
# Home Route
# ===============================
@app.route("/image", methods=["GET", "POST"])
def index():
    uploaded_file = None
    result_file = None
    count = None
    density = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            filename = secure_filename(file.filename)

            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # Run YOLO
            results = model.predict(upload_path, conf=0.25)
            
            # Count people
            count = len(results[0].boxes)

            # Save result image
            result_path = os.path.join(RESULT_FOLDER, filename)
            results[0].save(filename=result_path)

            uploaded_file = upload_path.replace("\\", "/")
            result_file = result_path.replace("\\", "/")

            # Density logic
            if count == 0:
                density = "No crowd"
            elif 1 <= count <= 9:
                density = "Low"
            elif 10 <= count <= 30:
                density = "Medium"
            else:
                density = "High"

            # ✅ INSERT INTO DATABASE
            insert_data(uploaded_file, result_file, count, density)

    return render_template(
        "index.html",
        uploaded_file=uploaded_file,
        result_file=result_file,
        count=count,
        density=density
    )


# ===============================
# 📊 Graph Analysis Route
# ===============================
@app.route("/graph")
def graph():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, count FROM crowd_data")
    #cursor.execute("SELECT MAX(count) FROM crowd_data")
    data = cursor.fetchall()

    print("DATA:", data)
    #total_images = cursor.fetchone()[0]
    total_images = len(data)
    total_people = sum([row[1] for row in data])
    #print("---",total_images)
      
    #max_count  = cursor.fetchall()
    conn.close()

    if data:
        ids = [str(row[0]) for row in data]
        counts = [row[1] for row in data]
        print(counts)
    
        #total_images = len(data)
        plt.figure()
        plt.bar(ids, counts)
        plt.xlabel("Image ID")
        plt.ylabel("People Count")
        plt.title("Crowd Count Analysis")
        plt.grid()

        graph_path = "static/results/graph.png"
        plt.savefig(graph_path)
        plt.close()
        image_count=ids
    else:
        graph_path = None

    return render_template("graph.html", graph_path=graph_path,total_images=total_images,total_people=total_people)


if __name__ == "__main__":
    app.run(debug=True)