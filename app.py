from flask import Flask, render_template, request
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")

# Folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_file = None
    result_file = None
    count = None
    density = None

    if request.method == "POST":
        file = request.files.get("image")  # ✅ matches HTML name="image"

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

            # Convert path for HTML (important)
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

    return render_template(
        "index.html",
        uploaded_file=uploaded_file,
        result_file=result_file,
        count=count,
        density=density
    )

if __name__ == "__main__":
    app.run(debug=True)