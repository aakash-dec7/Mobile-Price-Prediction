import pandas as pd
from Logger import logger
from src.s5_inference import Inference
from flask import Flask, render_template, request

app = Flask(__name__)

# Load Inference Model
inference = Inference()


@app.route("/", methods=["GET"])
def home_page():
    return render_template(
        "index.html", price_range_prediction="Price Range Prediction"
    )


@app.route("/predict", methods=["POST"])
def analyze():
    try:
        form = request.form

        # Creating a DataFrame from the form data
        input_data = pd.DataFrame(
            [
                {
                    "battery_power": int(form.get("battery_power", 1000)),
                    "blue": int(form.get("blue", 0)),
                    "clock_speed": float(form.get("clock_speed", 1.0)),
                    "dual_sim": int(form.get("dual_sim", 0)),
                    "fc": int(form.get("fc", 0)),
                    "four_g": int(form.get("four_g", 0)),
                    "int_memory": int(form.get("int_memory", 8)),
                    "n_cores": int(form.get("n_cores", 4)),
                    "pc": int(form.get("pc", 8)),
                    "px_height": int(form.get("px_height", 800)),
                    "px_width": int(form.get("px_width", 1280)),
                    "ram": int(form.get("ram", 1024)),
                    "sc_h": int(form.get("sc_h", 10)),
                    "sc_w": int(form.get("sc_w", 5)),
                    "three_g": int(form.get("three_g", 0)),
                    "touch_screen": int(form.get("touch_screen", 0)),
                    "wifi": int(form.get("wifi", 0)),
                }
            ]
        )

        # Prediction using the Inference model
        prediction = inference.predict(input_data=input_data.values)

        return render_template(
            "index.html", price_range_prediction="Price Range: " + str(prediction)
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template(
            "index.html",
            price_range_prediction="An error occurred. Please try again later.",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
