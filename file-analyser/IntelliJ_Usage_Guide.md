# IntelliJ IDEA Usage Guide - Medical RAG System

## Quick Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run in IntelliJ IDEA**
   - Open the project in IntelliJ IDEA
   - Open `simple-medical-rag-poc.py`
   - Run the file (it will automatically load `sample_medical_record.pdf`)

## Interactive Usage in IntelliJ IDEA

### Option 1: Use the Interactive Class

```python
from simple-medical-rag-poc import InteractiveMedicalRAG

# Create and setup the system
medical_rag = InteractiveMedicalRAG()

# Load the PDF
medical_rag.load_pdf("sample_medical_record.pdf")

# Query the system
answer = medical_rag.query("What medications is the patient taking?")
print(answer)

# Get document information
info = medical_rag.get_document_info()
print(info)
```

### Option 2: Use the Quick Start Function

```python
from simple-medical-rag-poc import quick_start

# This will automatically load the PDF and run example queries
medical_rag = quick_start()

# Now you can use medical_rag to ask your own questions
answer = medical_rag.query("Your question here")
print(answer)
```

## Sample Queries for the Medical Record

Try asking these questions about Sarah Johnson's medical record:

- "What medications is the patient taking?"
- "What was the diagnosis?"
- "What are the vital signs?"
- "What allergies does the patient have?"
- "What is the patient's medical history?"
- "What tests were performed?"
- "What was the treatment plan?"
- "Who is the attending physician?"

## IntelliJ IDEA Tips

1. **Python Console**: Use IntelliJ's Python console to interactively query the system
2. **Debugging**: Set breakpoints to understand how the RAG system works
3. **Multiple PDFs**: You can load additional PDF files by calling `medical_rag.load_pdf("path_to_other_pdf.pdf")`

## Troubleshooting

- If you get import errors, make sure all dependencies are installed: `pip install -r requirements.txt`
- If the PDF doesn't load, check that `sample_medical_record.pdf` is in the project directory
- For Unicode errors on Windows, the system now uses ASCII-only characters