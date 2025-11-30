import streamlit as st
import sqlite3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import time
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Text-to-SQL Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
    }
    .sql-output {
        background-color: #282c34;
        color: #abb2bf;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">Text-to-SQL Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fine-tuned Flan-T5 Model for Natural Language to SQL Translation</div>', unsafe_allow_html=True)
# ============================================================
# CONFIGURATION
# ============================================================
PROMPT_TEMPLATE = """Question: {question}
Schema:
{schema}
SQL:"""
# Model paths - adjust these to your setup
MODEL_PATH = "finetuned_flant5/final_model"
# Database examples with schemas
DATABASE_SCHEMAS = {
    "hospital_1": """Database: hospital_1
Tables:
- Physician(EmployeeID*, Name, Position, SSN)
- Department(DepartmentID*, Name, Head)
- Affiliated_With(Physician*, Department, PrimaryAffiliation)
- Procedures(Code*, Name, Cost)
- Trained_In(Physician*, Treatment, CertificationDate, CertificationExpires)
- Patient(SSN*, Name, Address, Phone, InsuranceID, PCP)
- Nurse(EmployeeID*, Name, Position, Registered, SSN)
- Appointment(AppointmentID*, Patient, PrepNurse, Physician, Start, End, ExaminationRoom)
- Medication(Code*, Name, Brand, Description)
- Prescribes(Physician*, Patient, Medication, Date, Appointment, Dose)
- Block(BlockFloor*, BlockCode)
- Room(RoomNumber*, RoomType, BlockFloor, BlockCode, Unavailable)
- On_Call(Nurse*, BlockFloor, BlockCode, OnCallStart, OnCallEnd)
- Stay(StayID*, Patient, Room, StayStart, StayEnd)
- Undergoes(Patient*, Procedures, Stay, DateUndergoes, Physician, AssistingNurse)""",
    
    "college_2": """Database: college_2
Tables:
- classroom(building*, room_number, capacity)
- department(dept_name*, building, budget)
- course(course_id*, title, dept_name, credits)
- instructor(ID*, name, dept_name, salary)
- section(course_id*, sec_id, semester, year, building, room_number, time_slot_id)
- teaches(ID*, course_id, sec_id, semester, year)
- student(ID*, name, dept_name, tot_cred)
- takes(ID*, course_id, sec_id, semester, year, grade)
- advisor(s_ID*, i_ID)
- time_slot(time_slot_id*, day, start_hr, start_min, end_hr, end_min)
- prereq(course_id*, prereq_id)""",
    
    "hr_1": """Database: hr_1
Tables:
- employees(employee_id*, first_name, last_name, email, phone_number, hire_date, job_id, salary, commission_pct, manager_id, department_id)
- departments(department_id*, department_name, manager_id, location_id)
- jobs(job_id*, job_title, min_salary, max_salary)
- locations(location_id*, street_address, postal_code, city, state_province, country_id)
- countries(country_id*, country_name, region_id)
- regions(region_id*, region_name)""",
    
    "store_1": """Database: store_1
Tables:
- artists(id*, name)
- sqlite_sequence(name, seq)
- albums(id*, title, artist_id)
- employees(id*, last_name, first_name, title, reports_to, birth_date, hire_date, address, city, state, country, postal_code, phone, fax, email)
- customers(id*, first_name, last_name, company, address, city, state, country, postal_code, phone, fax, email, support_rep_id)
- genres(id*, name)
- invoices(id*, customer_id, invoice_date, billing_address, billing_city, billing_state, billing_country, billing_postal_code, total)
- media_types(id*, name)
- tracks(id*, name, album_id, media_type_id, genre_id, composer, milliseconds, bytes, unit_price)
- invoice_items(id*, invoice_id, track_id, unit_price, quantity)
- playlist_track(playlist_id*, track_id)
- playlists(id*, name)"""
}
# Example questions for each database
EXAMPLE_QUESTIONS = {
    "hospital_1": [
        "How many physicians are there?",
        "List all departments and their heads",
        "Which patients have appointments scheduled?",
        "What medications are prescribed most frequently?",
        "Show all nurses on call in block floor 1"
    ],
    "college_2": [
        "How many students are enrolled?",
        "List all courses in the Computer Science department",
        "Which instructors teach the most courses?",
        "Show all students with GPA above 3.5",
        "What is the total budget of all departments?"
    ],
    "hr_1": [
        "How many employees work in each department?",
        "List all employees hired in 2023",
        "Which department has the highest average salary?",
        "Show all managers and their departments",
        "What jobs have salary range over 100000?"
    ],
    "store_1": [
        "How many albums does each artist have?",
        "List top 5 customers by total purchases",
        "Which genre has the most tracks?",
        "Show all employees and their managers",
        "What is the total revenue by country?"
    ]
}
# ============================================================
# LOAD MODEL (with caching)
# ============================================================
@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer"""
    try:
        with st.spinner("Loading fine-tuned model... This may take a moment."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
            model = model.to(device)
            model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure the model path is correct: `finetuned_flant5/final_model`")
        return None, None, None
# ============================================================
# SQL GENERATION FUNCTION
# ============================================================
def generate_sql(question, schema, model, tokenizer, device):
    """Generate SQL query from natural language question"""
    
    # Build prompt
    prompt = PROMPT_TEMPLATE.format(question=question, schema=schema)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            do_sample=False
        )
    
    generation_time = (time.time() - start_time) * 1000
    
    # Decode
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up SQL
    sql = sql.strip()
    if not sql.endswith(';'):
        sql += ';'
    
    return sql, generation_time
# ============================================================
# SCHEMA EXTRACTION FUNCTION
# ============================================================
def extract_schema_from_db(db_path):
    """Automatically extract schema from SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            return None, "No tables found in database"
        
        schema_parts = []
        db_name = Path(db_path).stem
        schema_parts.append(f"Database: {db_name}")
        schema_parts.append("Tables:")
        
        # Get columns for each table
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            # Format columns with primary key markers
            col_names = []
            for col in columns:
                col_name = col[1]
                is_pk = col[5] == 1
                if is_pk:
                    col_names.append(f"{col_name}*")
                else:
                    col_names.append(col_name)
            
            table_schema = f"- {table}({', '.join(col_names)})"
            schema_parts.append(table_schema)
        
        # Get foreign keys
        fk_lines = []
        for table in tables:
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            fks = cursor.fetchall()
            
            for fk in fks:
                from_col = fk[3]
                to_table = fk[2]
                to_col = fk[4]
                fk_line = f"FK {table}.{from_col} -> {to_table}.{to_col}"
                fk_lines.append(fk_line)
        
        if fk_lines:
            schema_parts.extend(fk_lines)
        
        conn.close()
        
        schema_text = "\n".join(schema_parts)
        return schema_text, None
        
    except Exception as e:
        return None, str(e)
# ============================================================
# EXECUTE SQL FUNCTION
# ============================================================
def execute_sql(sql, db_path):
    """Execute SQL query and return results"""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)
# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Configuration")
    
    # Database selection mode
    db_mode = st.radio(
        "Database Mode",
        ["Pre-defined Databases", "Upload Custom Database"],
        help="Choose to use pre-defined schemas or upload your own database"
    )
    
    if db_mode == "Pre-defined Databases":
        selected_db = st.selectbox(
            "Select Database",
            options=list(DATABASE_SCHEMAS.keys()),
            help="Choose which database to query"
        )
        custom_mode = False
    else:
        selected_db = "custom"
        custom_mode = True
        st.info("Upload your SQLite database below to auto-extract schema")
    
    st.markdown("---")
    
    # Show schema
    if not custom_mode:
        with st.expander("View Database Schema", expanded=False):
            st.code(DATABASE_SCHEMAS[selected_db], language="text")
    elif 'custom_schema' in st.session_state:
        with st.expander("View Extracted Schema", expanded=True):
            st.code(st.session_state.custom_schema, language="text")
    
    st.markdown("---")
    
    # Model info
    st.subheader("Model Information")
    st.info("""
    **Model:** Flan-T5 Base (Google)  
    **Base:** google/flan-t5-base  
    **Training:** 8,559 examples  
    **Performance:**
    - Exact Match: 67.3%
    - Execution Accuracy: 83.1%
    - Valid SQL: 89.8%
    """)
    
    st.markdown("---")
    
    # About
    st.subheader("About")
    st.markdown("""
    This demo showcases a fine-tuned Flan-T5 model 
    that translates natural language questions into 
    SQL queries for different database schemas.
    
    **Supported Databases:**
    - Hospital Management
    - University System
    - HR Management
    - Music Store
    - **Custom Upload:** Any SQLite database!
    """)
# ============================================================
# MAIN CONTENT
# ============================================================
# Load model
model, tokenizer, device = load_model()
if model is None:
    st.stop()
st.success("Model loaded successfully!")
# Database file upload or selection
st.subheader("Database Connection")
if not custom_mode:
    # Pre-defined database mode
    col1, col2 = st.columns([3, 1])
    with col1:
        db_file = st.file_uploader(
            f"Upload {selected_db}.sqlite database file",
            type=['sqlite', 'db'],
            help="Upload the corresponding SQLite database file",
            key="predefined_db_upload"
        )
    with col2:
        st.metric("Selected DB", selected_db)
    # Save uploaded file temporarily
    db_path = None
    current_schema = DATABASE_SCHEMAS[selected_db]
    
    if db_file is not None:
        db_path = f"temp_{selected_db}.sqlite"
        with open(db_path, "wb") as f:
            f.write(db_file.getbuffer())
        st.success(f"Database loaded: {selected_db}")
else:
    # Custom database mode
    st.info("Upload your SQLite database and we'll automatically extract the schema!")
    
    db_file = st.file_uploader(
        "Upload Custom SQLite Database",
        type=['sqlite', 'db'],
        help="Upload any SQLite database file",
        key="custom_db_upload"
    )
    
    db_path = None
    current_schema = None
    
    if db_file is not None:
        db_name = db_file.name.replace('.sqlite', '').replace('.db', '')
        db_path = f"temp_custom_{db_name}.sqlite"
        
        # Save file
        with open(db_path, "wb") as f:
            f.write(db_file.getbuffer())
        
        # Extract schema
        with st.spinner("Extracting database schema..."):
            schema, error = extract_schema_from_db(db_path)
            
            if error:
                st.error(f"Error extracting schema: {error}")
                db_path = None
            else:
                current_schema = schema
                st.session_state.custom_schema = schema
                
                # Show extracted info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Count tables
                    table_count = schema.count("\n- ")
                    st.metric("Tables Found", table_count)
                
                with col2:
                    # Count foreign keys
                    fk_count = schema.count("\nFK ")
                    st.metric("Foreign Keys", fk_count)
                
                with col3:
                    st.metric("Database", db_name)
                
                st.success(f"Schema extracted successfully from {db_file.name}!")
                
                # Show extracted schema
                with st.expander("View Extracted Schema", expanded=True):
                    st.code(schema, language="text")
st.markdown("---")
# Query interface
st.subheader("Ask a Question")
# Example questions (only for pre-defined databases)
if not custom_mode:
    st.markdown("**Try these example questions:**")
    cols = st.columns(5)
    for idx, example in enumerate(EXAMPLE_QUESTIONS[selected_db]):
        with cols[idx]:
            if st.button(f"Q{idx+1}", help=example, use_container_width=True):
                st.session_state.question = example
# Question input
question = st.text_area(
    "Enter your question in natural language:",
    value=st.session_state.get('question', ''),
    height=100,
    placeholder="e.g., How many records are in the main table?" if custom_mode else "e.g., How many physicians are there?"
)
# Show selected example question tooltip
if not custom_mode and question in EXAMPLE_QUESTIONS[selected_db]:
    st.info(f"Example question selected")
# Generate SQL button
if st.button("Generate SQL", type="primary", use_container_width=True):
    if not question:
        st.warning("Please enter a question first!")
    elif custom_mode and current_schema is None:
        st.warning("Please upload a database first!")
    elif not custom_mode and current_schema is None:
        st.warning("Schema not available. Please select a valid database!")
    else:
        with st.spinner("Generating SQL query..."):
            # Generate SQL
            sql, gen_time = generate_sql(
                question, 
                current_schema,
                model,
                tokenizer,
                device
            )
            
            # Store in session state
            st.session_state.generated_sql = sql
            st.session_state.generation_time = gen_time
            
            # Auto-execute if database is available
            if db_path:
                with st.spinner("Executing query..."):
                    df, error = execute_sql(sql, db_path)
                    st.session_state.execution_result = (df, error)
# Display results if SQL was generated
if 'generated_sql' in st.session_state:
    st.markdown("---")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Generation Time", f"{st.session_state.generation_time:.0f} ms")
    
    with col2:
        sql_length = len(st.session_state.generated_sql)
        st.metric("SQL Length", f"{sql_length} chars")
    
    with col3:
        display_db = selected_db if not custom_mode else "custom"
        st.metric("Database", display_db)
    
    # Generated SQL
    st.subheader("Generated SQL Query")
    st.code(st.session_state.generated_sql, language="sql")
    
    # Display execution results if available
    if 'execution_result' in st.session_state:
        df, error = st.session_state.execution_result
        
        if error:
            st.markdown(f'<div class="error-box"><strong>Execution Error:</strong><br>{error}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">Query executed successfully!</div>', unsafe_allow_html=True)
            
            # Display results
            st.subheader("Query Results")
            
            if len(df) == 0:
                st.info("Query returned no results.")
            else:
                # Show result count
                st.metric("Rows Returned", len(df))
                
                # Display dataframe
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"{selected_db}_results.csv",
                    mime="text/csv"
                )
    elif db_path:
        st.info("Query generated! Click 'Generate SQL' again to see execution results.")
    else:
        st.info("Upload a database file to execute the query and see results!")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Text-to-SQL Demo Application</strong></p>
    <p>Powered by Fine-tuned Flan-T5 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)