from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Load the TensorFlow/Keras model
model = load_model("app/cleaned_food_classifier_model.h5") 
class_labels = ['Food', 'Non_Food']

# Define a function to preprocess the image
def preprocess_image(file) -> np.ndarray:
    img = load_img(file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Define an endpoint to receive image uploads and make predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    with open("temp.jpg", "wb") as f:
        f.write(await file.read())
    
    # Preprocess the image
    img_array = preprocess_image("temp.jpg")
    
    # Make predictions using the loaded model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_label = class_labels[predicted_class]
    
    # Return the prediction as JSON response
    return JSONResponse(content={"prediction": class_label})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
    