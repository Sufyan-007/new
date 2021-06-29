from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse,HTMLResponse
import predict
import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
app = FastAPI()

@app.post('/files', response_class=HTMLResponse)
async def create_files(request: Request, files: List[bytes] = File(...)):
    img = np.fromstring(files[0], np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    predict.predict(img)
    return FileResponse("pp.jpg")
    
@app.get('/', response_class=HTMLResponse)
async def main(request: Request):
    content = """
    <html>
        <body>
            <h1> JIET Object Classification Model </h1>
            <form action="/files/" method="post" enctype="multipart/form-data">
            <input  name="files" type="file" multiple>
            <input type="submit">
            </form>
        </body>
    </html>
    """
    return content
