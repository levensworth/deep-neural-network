#this will be it

import numpy as np
import csv
class Config():
	#these are hyperparameters
	layers = [44,3,1] # this array contains the number of neurons in each layer
	reg_term = 0.01
	learnin_rate = 0.3
	momentum = 0.01

def build_model():
	#this functino is based on the config class which should be modified as best suited for the job
	model_weights = []
	model_biases = []

	for i in range(0,len(Config.layers)-1):
		# creates a matrix fro each leayer
		#model = np.append(model,np.random.rand(Config.layers[i],Config.layers[i+1]))
		model_weights.append( np.random.rand(Config.layers[i],Config.layers[i+1]) )

		model_biases.append(np.random.rand(1,Config.layers[i+1]))
		#print model
	return model_weights ,model_biases





# functions needed for the propagation

def tanh_prime(x):
	return 1 - np.tanh(x)**2
#this allows the function to be apply elemntwise
tanh_prime = np.vectorize(tanh_prime)


def cost(Yhat , y):
	result = 0.5 * np.sum(np.square(y - Yhat),axis =0)
	return result



#now we starte with the foward pass for predictions

def run(model, biases, X):
	# this is simply a foward pass
	# X is the input np matrix
	Z = X.dot(model[0]) + biases[0]
	A = np.tanh(Z)

	for i in range(1, len(Config.layers)-1):
		Z = A.dot(model[i]) + biases[i]
		A = np.tanh(Z)

	return A


# this function is called from the train function as it needs a predefined batch
def train_loop(model, biases, X, y,epochs):

	#X is the inpu
	#y is the output
	# these should be in a batch ready for process
	# we define the prev D vector outside the loop
	#for permanece issues
	prev_D = [0 for i in range(0, len(Config.layers) - 1)]

	for epoch in range(0,epochs):
		#first we do the foward prop
		Z_vector = []
		A_vector = []

		#this is for the input
		Z_vector.append(X.dot(model[0]) + biases[0])
		A_vector.append(np.tanh(Z_vector[0]))
		#for the rest of the layers
		for i in range(1, len(Config.layers)-1):
			Z_vector.append( A_vector[i-1].dot(model[i]) + biases[i] )
			A_vector.append(np.tanh(Z_vector[i]))

		#now the back prop
		D_vector = []


		D_vector.append( np.multiply(-(y - A_vector[len(Config.layers)-2]), tanh_prime(Z_vector[len(Config.layers)-2]) ))


		for i in range(len(Config.layers)-3, -1,-1):
			D_vector.append(np.multiply( D_vector[len(Config.layers)-3 - i].dot(model[i+1].T), tanh_prime(Z_vector[i])  ))

			# the -1 is to include the 0
			# as the last index is layers - 2 we put layers -3 to start from the previous to last
		# deltas are in reverse order

		if epoch == 0:
			for i in range(0, len(Config.layers) -2):
				prev_D[i] = D_vector[i]


		#after computing the deltas we calculat the derivatives
		for i in range(len(Config.layers) -2, 0,-1):
			djdw = ((A_vector[i-1].T).dot(D_vector[len(Config.layers)-2 - i] ))
			djdw += Config.reg_term * model[i]
			djdw = Config.learnin_rate * djdw
			djdw += (( 1 - Config.learnin_rate)* Config.momentum ) * prev_D[i]
			model[i] = model[i] - djdw

			#biases
			db = np.sum(D_vector[len(Config.layers) - 2 - i], axis = 0)
			biases[i] = biases[i] -  Config.learnin_rate * db

		#we update the prev_D vector
		for i in range(0, len(Config.layers) -2):
			prev_D[i] = D_vector[i]

		#now the input
		djdw = ((X.T).dot(D_vector[len(Config.layers)-2] ))
		djdw += Config.reg_term * model[0]
		model[0] = model[0] - Config.learnin_rate * djdw
		db = np.sum(D_vector[ len(Config.layers) - 2 ], axis = 0)
		biases[0] =biases[0] - Config.learnin_rate * db


		if epoch % 1000 == 0:
			print "cost"
			print cost(A_vector[-1],y)

	return model, biases





def train(model,biases, input_matrix, output, epochs, batch_size):

	size = input_matrix.shape


	if size[0] > 200:
		for i in range(0, size[0], batch_size):
			X = input_matrix[i:batch_size + i , :]
			y = output[i:batch_size + i , :]
			model, biases = train_loop(model, biases,X,y, epochs)
	else:
		X = input_matrix
		y = output
		model, biases = train_loop(model, biases,X,y, epochs)

	return model, biases
#------------------implementation-------------

def generate_data():
	X = np.matrix('1,1; 0,0; 1,0; 0,1')

	y = np.matrix('0;0;1;1')


	return X, y




def generate_train(file_name):
#lesto , tratalo con cuidado con el mismo criterio que hiciste con csv wrapper
	X = np.zeros((1,44))
	y = np.zeros((1,1))
	with open(file_name, 'rb') as csvfile:
		data_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in data_reader:


			X = np.append(X, generate_vector(row[2]).reshape((1,44)), axis = 0)
			review  = np.matrix([get_input(row[1], row[2])])
			y = np.append(y, review, axis = 0)




	return X,y



def get_input(title, genres):
	print " which is your rating " + title
	print " genres > "+ genres
	raw = raw_input()
	num = 0
	decimal = 0
	place = "num"
	result =  float(raw) / 10
	#the 10 is because it is the max rating
	return result


def generate_vector(genres):


	gen = ""
	state = "reading"
	result = np.zeros( 44)
	index = 0


	for i in genres:
		if i == ',':
			state = "wait"
			np.put(result, get_index(gen), 1)
			gen =""
		elif state == "reading":
			gen += i
		elif i == " " and state == "wait":
			state = "reading"

	return result


def get_data(file_name):
	X = np.zeros((1,44))

	with open(file_name, 'rb') as csvfile:
		i = 0

		data_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in data_reader:
			if i > 12:

				X = np.append(X, generate_vector(row[2]).reshape((1,44)), axis = 0)

			i +=1



	return X

def get_index(gen):
	gen_vector  = ["Comedy","Action","Adventure","Fantasy",
		"Sci-Fi","Drama","Shounen","Kids",
		"Romance","School","Slice of Life",
		"Hentai","Supernatural","Mecha",
		"Music","Historical","Magic","Ecchi",
		"Shoujo","Seinen","Sports","Mystery",
		"Super Power","Military","Parody",
		"Space","Horror","Harem","Demons",
		"Martial Arts","Dementia","Psychological",
		"Police","Game","Samurai","Vampire","Thriller",
		"Cars","Shounen Ai","None","Shoujo Ai","Josei",
		"Yuri","Yaoi"]

	return gen_vector.index(gen)




# from here it would be the main function




def main():
	model , biases = build_model()



	X, y = generate_train("anime_test.csv")

	new_mod, new_biases =train(model,biases, X, y, 10000,11)
	results  = run(new_mod, new_biases,get_data("anime.csv"))
	result  = results.tolist()

	i = 0
	with open("anime.csv", 'rb') as csvfile:
		data_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in data_reader:
			if i > 0 and  result[i-1][0] > 0.7 :

				print row[1] + " genres: "+ row[2] +" rating is > "
				print result[i-1][0]
				raw_input()

			i +=1








if __name__ == '__main__':
	main()
