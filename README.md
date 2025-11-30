---

# Text2SQL – Natural Language to SQL Converter

This project enables users to convert natural language queries into executable SQL commands using deep learning and transformer-based models.
It includes both a backend (Flask-based) and a frontend (Streamlit interface) for smooth interaction.

---

## Getting Started

Follow these steps to clone the repository, set up your environment, and run the Streamlit application.

### 1. Clone the Repository

```bash
gh repo clone AtharvaBOT7/SWM-Project
cd SWM-Project-main
```

---

### 2. Create a Virtual Environment

Create a Python virtual environment (Python 3.10 is recommended):

```bash
python3.10 -m venv venv
```

Activate the environment:

**For macOS/Linux:**

```bash
source venv/bin/activate
```

**For Windows (PowerShell):**

```bash
venv\Scripts\activate
```

---

### 3. Verify Python Version

Check that the correct Python version from your virtual environment is active:

```bash
which python
```

It should point to the path inside your `venv/` directory.

---

### 4. Install Dependencies

Install all required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 5. Run the Application

Start the Streamlit frontend:

```bash
streamlit run frontend.py
```

---

