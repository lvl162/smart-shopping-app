from flask import Flask, render_template, Response, url_for, redirect
from detect import detect
from flask import jsonify, request
import requests
import cv2
from flask_pymongo import PyMongo
from trans import *


app = Flask(__name__)


app.config['MONGO_DBNAME'] = 'CookyCooky'
# app.config['MONGO_URI'] = 'mongodb+srv://lvl162:1622000@cluster0.gabg8.gcp.mongodb.net/CookyCooky?retryWrites=true&w=majority'
app.config['MONGO_URI'] = 'mongodb://127.0.0.1:27017/CookyCooky'

mongo = PyMongo(app)


# camera = cv2.VideoCapture(0)  # use 0 for web camera
# print(camera.isOpened())
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

SEEN_OBJ = list()


def clear_seen(obj):
    obj = list()


@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='favicon.ico'))


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
    stream_url = "http://192.168.43.1:8080/video"
    # stream_url = "http://192.168.43.15:8000/stream.mjpg"
    weights_path = './best_5000_7.pt'
    return Response(detect(source=stream_url, conf_thres=0.7, weights=weights_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html')


@app.route('/pre_demo')
def pre_demo():
    id = request.args.get('id')
    if not id:
        id = 6969
    return render_template('ingredients-selection.html', ids=id)


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Video streaming home page."""
    # print(STRING)
    clear_seen(SEEN_OBJ)

    if request.method == 'GET':
        id = request.args.get('id')
        if not id:
            id = ''
        rm = request.args.get('rm')
        if not rm:
            rm = ''
        return render_template('demo.html', ids=id, rms=rm)
    if request.method == 'POST':
        pass


@app.route('/search')
def search():
    return render_template('search.html')


@app.route('/search3')
def search3():
    return render_template('search3.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/recipeDetails')
def details():
    id = request.args.get('id')
    if id:
        # recipe = mongo.db.recipes
        # s = recipe.find_one({'id': int(id)})

        return render_template('singleItem.html', ids=id)


def index():
    return render_template('index.html')


@app.route('/ingredients')
def show_ingredients():
    ids = request.args.get('id')
    ings = []
    rms = request.args.get('rm')
    if ids:
        id = ids.split(',')
        recipe = mongo.db.recipes
        for i in id:
            s = recipe.find_one({'id': int(i)})
            ings = ings + list(dict(s['ingredients']).keys())
        if rms:
            rm_arr = rms.split(',')
            rm_int = [int(i) for i in rm_arr]
            ings_tmp = list(ings)
            for index, ing in enumerate(ings):
                if index in rm_int:
                    ings_tmp.remove(ing)
            ings = list(ings_tmp)
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
            {'id': s['id'], 'name': s['name'], 'description': s['description'], 'ingredients': s['ingredients'], 'image': s['image'], 'likes': s['likes'], 'cooking_steps': s['cooking_steps'], 'ration': s['ration']})
    elif name:
        recipe = mongo.db.recipes
        ids = []
        for s in recipe.find():
            if convert(s['name']).find(convert(name)) >= 0:
                if s['id'] not in ids:
                    ids.append(s['id'])
                    output.append(
                        {'id': s['id'], 'name': s['name'], 'description': s['description'], 'ingredients': s['ingredients'], 'image': s['image'], 'likes': s['likes'], 'cooking_steps': s['cooking_steps'], 'ration': s['ration']})
    else:
        recipe = mongo.db.recipes
        for s in recipe.find():
            output.append(
                {'id': s['id'], 'name': s['name'], 'description': s['description'], 'ingredients': s['ingredients'], 'image': s['image'], 'likes': s['likes'], 'cooking_steps': s['cooking_steps'], 'ration': s['ration']})
    return jsonify({'result': output})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
