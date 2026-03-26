                                                                                                                                                      
"""
Production entrypoint for Zup Web.
                                                                                                                                                      
Usage (development): python serve.py
Usage (frozen):      zup-web.exe
"""
                                                                                                                                                      
from pathlib import Path
import uvicorn
                                                                                                                                                      
# NOTE: main.py already sets up sys.path to see ../zup-cli,
# so we don't need to repeat that here.
                                                                                                                                                      
if __name__ == "__main__":
    this_dir = Path(__file__).parent
                                                                                                                                                      
    # Run the FastAPI app defined in main.py
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8765,
        reload=False,   # production: disable autoreload
        workers=1,      # you can increase this if you want concurrency
        log_level="info",
        log_config=None,  # <--- ADD THIS LINE
    )
                                                                                                                                                      
