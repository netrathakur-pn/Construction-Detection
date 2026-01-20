# Unauthorized Construction Detection System

This project focuses on detecting **unauthorized construction activities** by comparing **before and after images** of a location using computer vision techniques.  
It is developed as a **final-year engineering project** with real-world relevance in **urban planning, smart cities, and government monitoring systems**.

---

## 📌 Problem Statement
Unauthorized construction is a major issue in urban areas, leading to:
- Illegal land usage
- Safety risks
- Poor city planning
- Environmental damage  

Manual inspection is inefficient and time-consuming.  
This project aims to **automate construction monitoring** using image comparison techniques.

---

## 🎯 Objectives
- Detect construction changes using **before & after images**
- Identify possible **unauthorized constructions**
- Provide a **web-based interface** for monitoring
- Reduce dependency on manual inspections

---

## 🧠 System Approach
1. Upload **before-construction image**
2. Upload **after-construction image**
3. Apply image processing techniques
4. Detect and highlight construction changes
5. Display results on the web interface

---

## 🛠 Technology Stack
- **Programming Language:** Python  
- **Framework:** Flask  
- **Libraries:** OpenCV, NumPy  
- **Frontend:** HTML, CSS  
- **Backend:** Python  
- **Version Control:** Git & GitHub  

---

## 📂 Project Structure
Construction-Detection/
│
├── app.py # Main Flask application
├── detection.py # Image processing & detection logic
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files
│
├── templates/ # HTML templates
├── Static/ # CSS & static assets
├── before_images/ # Images before construction
├── after_images/ # Images after construction


📸 Input & Output

Input:
 - Before construction image
 - After construction image

Output:
 - Highlighted image showing detected construction changes
 - Indication of possible unauthorized construction

🔮 Future Enhancements
 - Integration with GIS mapping
 - Drone or satellite image analysis
 - Deep Learning-based change detection
 - Automatic alerts for authorities
 - Cloud deployment for city-scale monitoring
