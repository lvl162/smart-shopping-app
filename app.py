from flask import Flask, render_template, Response, url_for, redirect
from detect import detect
from flask import jsonify, request
import requests
import cv2
from flask_pymongo import PyMongo
from trans import *


app = Flask(__name__)


app.config['MONGO_DBNAME'] = 'CookyCooky'
app.config['MONGO_URI'] = 'mongodb+srv://lvl162:1622000@cluster0.gabg8.gcp.mongodb.net/CookyCooky?retryWrites=true&w=majority'
mongo = PyMongo(app)


# camera = cv2.VideoCapture(0)  # use 0 for web camera
# print(camera.isOpened())
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

SEEN_OBJ = list()


@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='favicon.ico'))


def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            print('no')
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # print("yes\n")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/post_seen', methods=['GET', 'POST'])
def post_seen():
    if request.method == 'GET':
        dic = {'mi_goi': "Mì gói", 'xuc_xich': 'Xúc xích', 'trung_ga': "Trứng gà",
               'le': "Lê", 'tao': "Táo", "cam": "Cam", 'sua_chua': "Sữa chua"}
        obj_x = request.args.get('obj')
        obj_name = dic[obj_x]
        if obj_name not in SEEN_OBJ:
            SEEN_OBJ.append(obj_name)
        # print(SEEN_OBJ)
    return jsonify({"a": "a"})


@app.route('/get_seen', methods=['GET', 'POST'])
def get_seen():
    # print(SEEN_OBJ)
    return jsonify({"get_seen": list(SEEN_OBJ)})


@app.route('/video_feed1')
def video_feed1():
    # Video streaming route. Put this in the src attribute of an img tag
    stream_url = "http://192.168.0.105:8080/video"
    weights_path = './best_500_7.pt'
    return Response(detect(source=stream_url, conf_thres=0.85, weights=weights_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "404 not found"})


@app.route('/pre_demo')
def pre_demo():
    id = request.args.get('id')
    if not id:
        id = 6969
    return render_template('pre_demo.html', ids=id)


@app.route('/demo')
def demo():
    """Video streaming home page."""
    # print(STRING)
    id = request.args.get('id')
    if not id:
        id = ''
    return render_template('demo.html', ids=id)


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/ingredients')
def show_ingredients():
    ids = request.args.get('id')
    ings = []
    if ids:
        id = ids.split(',')
        recipe = mongo.db.recipes
        for i in id:
            s = recipe.find_one({'id': int(i)})
            ings = ings + list(dict(s['ingredients']).keys())

    return jsonify({'result': ings})


@app.route('/recipes', methods=['GET'])
def get_all_recipes():
    output = []
    id = request.args.get('id')
    name = request.args.get('name')
    if id:
        recipe = mongo.db.recipes
        s = recipe.find_one({'id': int(id)})
        output.append(
            {'id': s['id'], 'name': s['name'], 'description': s['description'], 'ingredients': s['ingredients']})
    elif name:
        recipe = mongo.db.recipes
        for s in recipe.find():
            if convert(s['name']).find(convert(name)) >= 0:
                output.append(
                    {'id': s['id'], 'name': s['name'], 'description': s['description'], 'ingredients': s['ingredients']})
    else:
        recipe = mongo.db.recipes
        for s in recipe.find():
            output.append(
                {'id': s['id'], 'name': s['name'], 'description': s['description'], 'ingredients': s['ingredients']})
    return jsonify({'result': output})


if __name__ == '__main__':
    app.run(debug=True)
