from fastapi import FastAPI, File, Request
from fastapi.responses import FileResponse,HTMLResponse
import predict
import numpy as np
from typing import List
app = FastAPI()

@app.get('/', response_class=HTMLResponse)
async def main(request: Request):
    #displaying webpage to get image input
    page = """
    <html>
        <body>
            <h1>CELEBRITY LOOKALIKE DETECTOR</h1>
            <form action="/files/" method="post" enctype="multipart/form-data">
            <input  name="files" type="file" multiple>
            <input type="submit">
            </form>
        </body>
    </html>
    """
    return page

@app.post('/files', response_class=HTMLResponse)
async def display(request: Request, files: List[bytes] = File(...)):
    #recivied image
    img = np.fromstring(files[0], np.uint8)
    #Getting the prediction
    predict.predict(img)
    return FileResponse("predicted.jpg")
    
