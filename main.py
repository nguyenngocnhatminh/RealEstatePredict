from flask import Flask, jsonify, render_template
import utils

app = Flask(__name__)

@app.route("/predict/lstm/<int:region_id>", methods = ["GET"])
def lstmPrediction(region_id):
    data = utils.lstmPrediction(region_id)
    return jsonify({"data" : data})

@app.route("/predict/moving_avage/<int:region_id>", methods = ["GET"])
def mvaPrediction(region_id):
    data = utils.Moving_Avage(region_id)
    return jsonify({"data" : data})

@app.route("/predict/prophet/<int:region_id>" , methods=["GET"])
def prophetPrediction(region_id):
   data = utils.Prophet(region_id)
   return jsonify({"data" : data})

@app.route("/region", methods=["GET"])
def getALlRegion():
    data = utils.GetAllRegion()
    print(data)
    return jsonify({"region": data})

if __name__ == "__main__":
    app.run()