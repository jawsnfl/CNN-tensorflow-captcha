
import numpy as np 
import tensorflow as tf 

from PIL import Image

import matplotlib.pyplot as plt 
import os
from random import choice

#test image
targetDir = "bingo2"
img_list = []

def gen_list():
	for parent, dirnames, filenames in os.walk(targetDir):
		for filename in filenames:
		    if(filename.find(".png") != -1):
		    	#print("fimame :"+ filename)
		    	img_list.append(filename.replace(".png",""))
	return img_list
		#print("parent is : "+parent)
		#print("filename is ; "+filename)
		#filename is ; ZXMY.png
#x = random.randint(0, len(img_list)-1)
#print(img_list[x])

img_list = gen_list()
def getImage():
	img = choice(img_list)
	img_path = os.path.join(targetDir,img+'.png')
	#print("OPEN", img_path)
	#captcha_image = Image.open(targetDir+"\\"+img+".png")
	captcha_image = Image.open(img_path)
	captcha_image = np.array(captcha_image)
	#print("sdfdsdf---", captcha_image.shape)

	return img, captcha_image
'''text, image = gen_captcha_text_and_image()
print("验证码 2：", type(image))
print("验证码图像channel ：", image.shape)'''
# 图像大小
IMAGE_HEIGHT = 25
IMAGE_WIDTH = 52
MAX_CAPTCHA = 4
print("验证码文本最长字符数", MAX_CAPTCHA)

def convert2gray(img):
	if len(img.shape) > 2 :
		gray = np.mean(img, -1)

		return gray
	else:
		return img


#文本转向量

#char_set = number + alphabet +ALPHABET + ['_']
#CHAR_SET_LEN = len(char_set)
CHAR_SET_LEN = 37
def text2vec(text):
	text_len = len(text)
	if text_len > MAX_CAPTCHA:
		raise ValueError('验证码最长4个字符')

	vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)

	def char2pos(c):
		if c == '_':
			k = 62
			return k
		k = ord(c) - 48
		if k > 9 :
			k = ord(c) - 55
			'''if k > 35 :
				k = ord(c) - 61
				if k > 61 :
					raise ValueError('No Map')'''

		return k
	for i, c in enumerate(text):
		idx = i * CHAR_SET_LEN + char2pos(c)
		vector[idx] = 1
	return vector


#向量转回文本
def vec2text(vec):
	char_pos = vec.nonzero()[0]
	text = []
	for i, c in enumerate(char_pos):
		char_at_pos = i
		char_idx = c % CHAR_SET_LEN
		if char_idx < 10:
			char_code = char_idx + ord('0')
		elif char_idx <36:
			char_code = char_idx - 10 + ord('A')
		'''elif char_idx <62:
			char_code = char_idx - 36 + ord('a')
		elif char_idx == 62:
			char_code = ord('_')
		else:
			raise ValueError('error')'''
		text.append(chr(char_code))
	return "".join(text)


def get_next_batch(batch_size = 128):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
	batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

	for i in range(batch_size):
		text, image = wrap_gen_captcha_text_and_image()
		image = convert2gray(image)

		batch_x[i,:] = image.flatten() / 255
		batch_y[i,:] = text2vec(text)


	return batch_x, batch_y


with tf.name_scope('inputs'):
	X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH], name = 'x_input')
	Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN], name = 'y_input')

keep_prob = tf.placeholder(tf.float32)

#define CNN
def crack_captcha_cnn(w_alpha =  0.01, b_alpha = 0.1):
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])


	# 3 conv layer
	w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
	b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))

	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1,1,1,1], padding='SAME'),b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv1 = tf.nn.dropout(conv1, keep_prob)

	w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
	b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'),b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv2 = tf.nn.dropout(conv2, keep_prob)

	w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
	b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	# Fully connected layer
	#w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
	w_d = tf.Variable(w_alpha*tf.random_normal([4*7*64, 1024]))
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
	b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
	out = tf.add(tf.matmul(dense, w_out), b_out)
	#out = tf.nn.softmax(out)
	return out



def crack_captcha(captcha_image):
	output = crack_captcha_cnn()

#output--> Tensor("Add_1:0", shape=(?, 252), dtype=float32)
	print("hyyy,", output)
 
	saver = tf.train.Saver()
	print("saver 1- : ", saver)
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('.'))

		print(output.shape, "restore ??")
		print(captcha_image.shape, "restore ---??")
		print("hyyy,", "saver afterr")
		print("saver : ", saver)
 
		predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
		text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
 
		text = text_list[0].tolist()
		vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
		i = 0
		for n in text:
				vector[i*CHAR_SET_LEN + n] = 1
				i += 1
		return vec2text(vector)

if __name__=='__main__':

	text, image = getImage()

	plt.imshow(image)
	plt.show()
	
	image = convert2gray(image)
	plt.imshow(image)
	plt.show()
	image = image.flatten() / 255


	predict_text = crack_captcha(image)
	print("正确: {}  预测: {}".format(text, predict_text))

 
