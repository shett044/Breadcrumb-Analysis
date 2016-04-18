from flask import Flask
from flask import render_template, jsonify, request, Flask,  redirect, url_for
import pandas as pd
import numpy as np
import json
app = Flask(__name__)

FILENAME = 'templates/visit_seq.csv'

# FIELDNAME = ("dd_reportingdate", "dd_forecastdate", "dim_partid", "dd_level2", "ct_salesquantity",
#              "ct_forecastquantity", "ct_lowpi", "ct_highpi", "ct_mape", "dd_lastdate", "dd_holdoutdate",
#              "dd_forecastsample", "dd_forecasttype", "dd_forecastrank", "dd_forecastmode", "dd_companycode")

part_id = ""
reporting_date = ""
forecast_type = ""
level2= ""
data = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/breadcrumb/loadData")
def data_pre_process():
    """
    Get the data preloaded
    :return: Data frame in JSON
    """

    return pd.DataFrame(data["title"].unique(), columns = ['title']).to_csv(index=False, header=None)


@app.route("/breadcrumb/getPartList/<string:title>")
def level_2_select_filter_part(title):
    """
    Returns Part list associated to  level 2 (Comop)
    :param level2:
    :return: JSON of unique parts associated to  level 2
    """
    global data
    return data.loc[data.title == title, ["seq", "count"]].to_csv(index=False, header=None)


@app.route('/breadcrumb/filter',methods=["GET"])
def filterData_ajax():
    """
    Filter according to the part, reporting and ForecastType
    :return:
    """
    global data, title
    title = request.args.get('title', title, type=str)
    return data[data.title == title, ["seq", "count"]].to_csv(index=False, header=None)

# @app.route("/breadcrumb/filter/<string:part_id>")
# def data_filter_part(part_id):
#     """
#     Filter dara according to the part_id
#     :param part_id:
#     :return: JSON of part and reporting_id
#     """
#     global data
#     # part_id = request.args.get('part_id', "", type=str)
#     return


# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({'error': 'Not found resource'})


if __name__ == "__main__":
    data = pd.read_csv(FILENAME)
    data = data[data["count"]>1]
    data=data[~data['seq'].str.contains('Not Set')]
    # data['dim_partid'] = data.dim_partid.astype(str)
    app.run(host='0.0.0.0', port=5000,  debug=True)
