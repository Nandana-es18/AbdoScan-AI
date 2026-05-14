from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
from inference_utils import load_inference_model, process_and_visualize
from generate_report import generate_full_clinical_report

import tempfile

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories - Use System Temp to avoid triggering Live Server reloads
BASE_TEMP_DIR = tempfile.mkdtemp(prefix="neuroscan_")
UPLOAD_DIR = os.path.join(BASE_TEMP_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_TEMP_DIR, "test_results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mount the static directory so the frontend can access the images via URL
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

print(f"Using temporary directories:\nUploads: {UPLOAD_DIR}\nResults: {RESULTS_DIR}")

# Load Model once on startup
print("Loading model...")
try:
    model = load_inference_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

@app.post("/analyze")
async def analyze_scan(
    file: UploadFile = File(...),
    modality: str = Form(...)
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension == '.gz' and file.filename.endswith('.nii.gz'):
        file_extension = '.nii.gz'
        
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process
        full_file_path = os.path.abspath(file_path)
        print(f"Processing {file.filename} as {modality}...")
        
        result_path = process_and_visualize(
            model, 
            full_file_path, 
            modality, 
            output_dir=RESULTS_DIR
        )
        
        return FileResponse(result_path, media_type="image/png", filename=os.path.basename(result_path))

    except Exception as e:
        print(f"Error processing scan: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # cleanup upload? maybe keep for debugging for now
        pass

@app.post("/diagnose")
async def diagnose_scan(
    file: UploadFile = File(...),
    organ: str = Form(...),  # Expected "liver" or "kidney"
    modality: str = Form(...) # Expected "ct" or "mri"
):
    # Save uploaded file safely
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension == '.gz' and file.filename.endswith('.nii.gz'):
        file_extension = '.nii.gz'
        
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        full_file_path = os.path.abspath(file_path)
        print(f"Generating Detailed Diagnostic Report for {organ}...")
        
        # 1. Run inference & generate the 3-panel XAI / AI image
        report_data = generate_full_clinical_report(
            image_path=full_file_path, 
            organ=organ,
            modality=modality,
            output_dir=RESULTS_DIR
        )
        
        # 2. Add full URL for frontend access
        # report_data["report_image_path"] holds absolute local path
        filename = os.path.basename(report_data["report_image_path"])
        image_url = f"http://localhost:8000/results/{filename}"
        
        return JSONResponse(content={
            "image_url": image_url,
            "organ_analyzed": organ.capitalize(),
            "has_tumor": report_data["has_tumor"],
            "radiomics": report_data.get("stats", {})
        })

    except Exception as e:
        print(f"Error processing diagnosis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
