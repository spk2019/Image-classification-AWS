from flask import Flask,request,render_template
from preprocessing import transform_image, predict_result
import torch
from PIL import Image

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            img_path = request.files['file'].stream
            image = Image.open(img_path)
            transformed_img =  transform_image(image)
            result = predict_result(transformed_img)
            if result == 0:
                result = "Audi"
            elif result == 1:
                result = "Toyota"
            print(result)
            return render_template("result.html", predictions=str(result))

    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code



if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True,port=5000)