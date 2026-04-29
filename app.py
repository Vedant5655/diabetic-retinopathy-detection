from flask import Flask, render_template, request, redirect, session
import sqlite3
import os
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)
app.secret_key = "secret123"

# -------- LOAD MODEL --------
model = tf.keras.models.load_model("dr_model.keras")

# -------- CROP FUNCTION --------
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        if img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0] == 0:
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.dstack((img1,img2,img3))
        return img

# -------- FINAL RETINA VALIDATION --------
def is_retina_image(img):
    img = cv2.resize(img, (224,224))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Brightness
    mean_intensity = np.mean(gray)

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (224 * 224)

    # Color dominance (KEY)
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    red_ratio = np.mean(r) / (np.mean(g) + 1e-5)

    # Final condition
    if (
        mean_intensity < 30 or mean_intensity > 220 or
        edge_density < 0.005 or
        red_ratio < 1.2
    ):
        return False

    return True

# -------- DATABASE --------
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            name TEXT,
            email TEXT,
            phone TEXT,
            age INTEGER,
            gender TEXT,
            diabetic TEXT,
            duration TEXT,
            password TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# -------- LOGIN --------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_input = request.form["user_input"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE (email=? OR name=? OR username=?) AND password=?",
            (user_input, user_input, user_input, password)
        )

        user = cursor.fetchone()
        conn.close()

        if user:
            session["user"] = user[2]
            return redirect("/dashboard")
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html", error="")

# -------- REGISTER --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = (
            request.form["username"],
            request.form["name"],
            request.form["email"],
            request.form["phone"],
            request.form["age"],
            request.form["gender"],
            request.form["diabetic"],
            request.form["duration"],
            request.form["password"]
        )

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO users 
                (username, name, email, phone, age, gender, diabetic, duration, password)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)

            conn.commit()
            conn.close()
            return redirect("/")
        except:
            return render_template("register.html", error="User already exists")

    return render_template("register.html")

# -------- DASHBOARD --------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("dashboard.html", username=session["user"], error="No file selected")

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template("dashboard.html", username=session["user"], error="Invalid file format")

        filepath = os.path.join("static/uploads", file.filename)
        file.save(filepath)

        # -------- LOAD IMAGE --------
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # -------- VALIDATION --------
        if not is_retina_image(img):
            return render_template(
                "dashboard.html",
                username=session["user"],
                error="Please upload a valid retina (fundus) image"
            )

        # -------- PREPROCESS --------
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (224,224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # -------- PREDICTION --------
        pred = model.predict(img)[0][0]
        print("Prediction:", pred)

        # -------- FINAL LOGIC --------
        if pred > 0.75:
            result = "No DR"
            confidence = pred * 100
        elif pred < 0.25:
            result = "DR"
            confidence = (1 - pred) * 100
        else:
            result = "Uncertain"
            confidence = pred * 100

        confidence = round(confidence, 2)

        return render_template("result.html",
                               result=result,
                               confidence=confidence,
                               image_path=filepath)

    return render_template("dashboard.html", username=session["user"])

# -------- LOGOUT --------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# -------- RUN --------
if __name__ == "__main__":
    app.run(debug=True)