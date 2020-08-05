import os,json,base64
import numpy as np
import tensorflow as tf
from flask import Flask , render_template , request
from keras.models import load_model
from keras.preprocessing import image

# 解决cuDNN failed to initialize
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


modelDir = "logs/001"
modelName = [x for x in sorted(os.listdir(modelDir)) if x.endswith(".h5")][-1]
imgPath = "pic.jpg"

graph_class = tf.Graph()
sess_class = tf.Session(graph=graph_class)

with sess_class.as_default():
	with graph_class.as_default():
		model = load_model(modelDir+"/"+modelName)

def predict_class(imgPath):
	global graph_class,model
	with sess_class.as_default():
		with graph_class.as_default():
			img = image.load_img(imgPath, target_size=(150, 150))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			y = model.predict_classes(x)
			if(int(np.squeeze(y))==0):
				return "该图片是小猫！"
			else:
				return "该图片是小狗!"



app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/postimg',methods = ['POST','GET'])
def getimg():
	if request.method =='POST':
		img = request.form['imgMsg']
		data = json.loads(img)
		for img_data in data:
			img_base64 = str(img_data['base64']);
		img_base64= img_base64.replace("data:image/jpeg;base64,","");
		fh = open("pic.jpg","wb")
		fh.write(base64.b64decode(img_base64))
		fh.close();
		result = predict_class(imgPath)
		return '%s' %result
	else:
		img = request.args.get('imgMsg')
		return 'success! %s' %img


if __name__ == '__main__':
	app.run(debug = True) 