import json
from flask import jsonify, request
from model import eeg_model



def eeg_controller(app):
    @app.route('/mental-connect/algorithm/api/eeg/anxiety', methods=['POST'])
    def eeg_anxiety():
        try:
            data = request.get_json()
            eeg_fp1_datas = data.get('fp1')
            eeg_fp2_datas = data.get('fp2')
            ans = eeg_model.eeg_anxiety(eeg_fp1_datas, eeg_fp2_datas)
            tensor_list = ans.tolist()
            json_data = json.dumps(tensor_list)
            responseDate = {
                "code": 0,
                "res": json_data
            }
            return jsonify(responseDate)
        except Exception as e:
            return jsonify(code = 1, message=str(e))