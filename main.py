from fastapi import FastAPI, UploadFile, File
from PIL import Image
from classfier import classify, result


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World This is Abhishek Sahu"}

@app.post("/image")
async def get_image(file : UploadFile = File(...)):
    im = Image.open(file.file)
    print(im.format)
    #classify(im)
    r = classify(im)
    return {"Class" : r.Class , "Confidence score" : str(r.Confidence)} 


