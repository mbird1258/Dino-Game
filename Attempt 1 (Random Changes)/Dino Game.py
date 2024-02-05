import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tkinter as tk
import time
import random
from datetime import datetime

#details
print("neural network visualisation key:")
print("======================================================")
print("IN:")
print("player height above ground")
print("next obj dist")
print("screen scroll speed")
print("next obj width")
print("nexrt obj height")
print("\nOUT:")
print("nothing")
print("jump")
print("drop\n\n")

#classes
class game:
	def __init__(self):
		#variables
		self.ScrollSpeed = 6

		self.Canvas_Width = 1200
		self.Canvas_Height = 600
		self.Ground_Height = 100
		self.Ground_Y_Value = self.Canvas_Height - self.Ground_Height

		self.Obstacles = np.array([[-1, -1, 0, 0]]) #[[tkinter shape, x, width, height], ...] #dummy 'object' that gets insta deleted
		self.Dist_Between_Obstacles = np.array([100, 250])
		self.Last_Pipe_x = 200
		self.Obstacle_Width = np.array([30, 80])
		self.Obstacle_Height = np.array([50, 80])
		self.distance = 0

		self.alive_players = np.array([], dtype = int)
		self.alive_neural_networks = np.array([])

		#create the window
		self.root = tk.Tk()
		self.canvas = tk.Canvas(self.root, bg="sky blue", height=self.Canvas_Height, width=self.Canvas_Width)
		self.root.title("Dinosaur game")
		self.canvas.pack()
		
		#create the ground
		self.ground = self.canvas.create_rectangle((0, self.Canvas_Height), (self.Canvas_Width, self.Ground_Y_Value), fill="lawn green", outline="")
		for i in range(10):
			self.create_obstacle()

	def create_obstacle(self):
		'''
		x = left side
		'''
		new_x = (np.random.random() * (self.Dist_Between_Obstacles[1] - self.Dist_Between_Obstacles[0]) + self.Dist_Between_Obstacles[0]) + self.Last_Pipe_x + self.ScrollSpeed * 5 #random value between the upper and lower number of self.Dist_Between_Obstacles + multiple of scroll speed(account for increasing speed) + last pipe's right side
		new_width = np.random.random() * (self.Obstacle_Width[1] - self.Obstacle_Width[0]) + self.Obstacle_Width[0]
		new_height = np.random.random() * ((-1/5 * new_width + self.Obstacle_Height[1] + self.Obstacle_Width[0]/5) - self.Obstacle_Height[0]) + self.Obstacle_Height[0] #https://www.desmos.com/calculator/skoed8f8a6

		self.Obstacles = np.append(self.Obstacles, [[self.canvas.create_rectangle((new_x - self.distance, self.Ground_Y_Value), (new_x + new_width - self.distance, self.Ground_Y_Value - new_height), fill="brown", outline=""), new_x - self.distance, new_width, new_height]], axis = 0)

		self.Last_Pipe_x = new_x + new_width

	def refresh(self):
		for obj in self.Obstacles[:, 0].astype(int):
			self.canvas.move(obj, -self.ScrollSpeed, 0)
		screen.Obstacles[:, 1] -= self.ScrollSpeed
		
		self.ScrollSpeed += 0.02/self.ScrollSpeed

		self.distance += self.ScrollSpeed

		self.remove_out_of_bounds()

		self.canvas.update()

	def remove_out_of_bounds(self):
		for obj in self.Obstacles[self.Obstacles[:,1] + self.Obstacles[:,2] < 0]:
			self.canvas.delete(int(obj[0]))
			self.create_obstacle()
		self.canvas.delete(self.Obstacles[0][0])
		self.Obstacles = np.delete(self.Obstacles, np.where(self.Obstacles[:,1] + self.Obstacles[:,2] < 0)[0], 0)

	def recreate(self):
		self.ScrollSpeed = 6
		self.Obstacles = np.array([[-1, -1, 0, 0]])
		self.distance = 0
		self.Last_Pipe_x = 200
		self.alive_players = np.array([], dtype = int)
		self.alive_neural_networks = neural_networks
		
		self.canvas.delete("all")
		self.canvas.pack()
		
		self.ground = self.canvas.create_rectangle((0, self.Canvas_Height), (self.Canvas_Width, self.Ground_Y_Value), fill="lawn green", outline="")
		
		for i in range(10):
			self.create_obstacle()

class player:
	def __init__(self, parent, number):
		self.Player_Y_Pos = parent.Ground_Y_Value
		self.Player_X_Pos = np.random.random() * 100 + 50 #random value between 50 and 150
		self.Player_Y_Vel = 0
		self.Score = 0
		self.Player_Jump_Height = 15
		self.Player_Dimensions = [30, 50]
		self.Dead = False
		self.Grounded = False
		self.Gravity = 1
		self.index = number
		self.player = parent.canvas.create_rectangle((self.Player_X_Pos, self.Player_Y_Pos), (self.Player_X_Pos + self.Player_Dimensions[0], self.Player_Y_Pos - self.Player_Dimensions[1]), fill="green", outline="red")

		parent.alive_players = np.append(parent.alive_players, self.index)

	def gravity(self):
		if self.Grounded == False:
			self.Player_Y_Vel -= self.Gravity

	def jump(self):
		if self.Grounded == True:
			self.Player_Y_Vel = self.Player_Jump_Height

	def drop(self):
		self.Player_Y_Vel = -20
		pass

	def move(self, parent):
		if -1 * self.Player_Y_Vel < parent.Ground_Y_Value - self.Player_Y_Pos: #when moving down, if you'd move past the ground, move to the ground instead.
			parent.canvas.move(self.player, 0, -1 * self.Player_Y_Vel)
			self.Player_Y_Pos -= self.Player_Y_Vel
			self.Grounded = False
		else:
			parent.canvas.move(self.player, 0, parent.Ground_Y_Value - self.Player_Y_Pos)
			self.Player_Y_Pos += parent.Ground_Y_Value - self.Player_Y_Pos
			self.Grounded = True

	def collision_detection(self, parent):
		global players
		for item in parent.Obstacles[:,0].astype(int):
			'''
			x1, y1 are bottom left corner
			x2, y2 are top right corner
			'''
			x1, y2, x2, y1 = parent.canvas.coords(item)

			for obj in parent.canvas.find_overlapping(x1, y1, x2 + parent.ScrollSpeed, y2 - self.Player_Y_Vel): #refresh(move) --> detect death, so extend hitbox(post-move) to the right(effectively extending the hitbox of the player to the left) and up(on negative y_vel) by move amount so teleporting through objects isn't possible
				if obj == self.player:
					self.Dead = True
					
					self.Score += parent.distance + self.Player_X_Pos + (x1 - (self.Player_X_Pos + self.Player_Dimensions[0])) #distance travelled compensating for start position and overshoot()

					parent.canvas.delete(self.player)
					parent.alive_players = parent.alive_players[parent.alive_players != self.index]
					parent.alive_neural_networks = parent.alive_neural_networks[parent.alive_neural_networks != neural_networks[self.index]]

	def respawn(self, parent):
		self.player = parent.canvas.create_rectangle((self.Player_X_Pos, self.Player_Y_Pos), (self.Player_X_Pos + self.Player_Dimensions[0], self.Player_Y_Pos - self.Player_Dimensions[1]), fill="green", outline="red")

		parent.alive_players = np.append(parent.alive_players, self.index)

class neural_network:
	def __init__(self, parent1, parent2):
		'''
		L# stands for layer #
		L## stands for in between layer(weights)
		LB# stands for layer bias
		
		inputs: grounded?, 1&2 dist, 1&2 width, 1&2 height
		'''
		next_two_obstacles = [parent2.Obstacles[parent2.Obstacles[:, 1] + parent2.Obstacles[:, 2] > parent1.Player_X_Pos][0], parent2.Obstacles[parent2.Obstacles[:, 1] + parent2.Obstacles[:, 2] > parent1.Player_X_Pos][1]]

		self.nodes = [5, 5, 5, 3]
		self.L1 = np.array([[-parent1.Player_Y_Pos+parent2.Ground_Y_Value, next_two_obstacles[0][1] - parent1.Player_X_Pos, screen.ScrollSpeed*10, next_two_obstacles[0][2], next_two_obstacles[0][3]]])
		
		self.L12 = np.random.randn(self.nodes[0], self.nodes[1]) * np.sqrt(2/self.nodes[0])
		self.LB2 = np.zeros((1, self.nodes[1]))
		
		self.L2 = np.zeros((1, self.nodes[1]))
		
		self.L23 = np.random.randn(self.nodes[1], self.nodes[2]) * np.sqrt(2/self.nodes[1])
		self.LB3 = np.zeros((1, self.nodes[2]))
		
		self.L3 = np.zeros((1, self.nodes[2]))
		
		self.L34 = np.random.randn(self.nodes[2], self.nodes[3]) * np.sqrt(2/self.nodes[2])
		self.LB4 = np.zeros((1, self.nodes[3]))
		
		self.L4 = np.zeros((1, self.nodes[3]))
		
		self.index = parent1.index
		self.parent = parent1.player

		parent2.alive_neural_networks = np.append(parent2.alive_neural_networks, self)

	def reproduce(self, father, mother):
		#get average model params of both parents
		basis_L12 = (father.L12 + mother.L12)/2
		basis_LB2 = (father.LB2 + mother.LB2)/2

		basis_L23 = (father.L23 + mother.L23)/2
		basis_LB3 = (father.LB3 + mother.LB3)/2

		basis_L34 = (father.L34 + mother.L34)/2
		basis_LB4 = (father.LB4 + mother.LB4)/2

		#make slight changes to these model params
		scale = 0.02
		self.new_L12 = basis_L12 * (1 + np.random.normal(0, scale, (self.nodes[0], self.nodes[1]))) + np.random.normal(0, scale, ((self.nodes[0], self.nodes[1]))) #basis increased by x% and a flat y, x and y are seperate normal distribution random numbers
		self.new_LB2 = basis_LB2 * (1 + np.random.normal(0, scale, (1, self.nodes[1]))) + np.random.normal(0, scale, (1, self.nodes[1])) #basis increased by x% and a flat y, x and y are seperate normal distribution random numbers

		self.new_L23 = basis_L23 * (1 + np.random.normal(0, scale, (self.nodes[1], self.nodes[2]))) + np.random.normal(0, scale, ((self.nodes[1], self.nodes[2]))) #basis increased by x% and a flat y, x and y are seperate normal distribution random numbers
		self.new_LB3 = basis_LB3 * (1 + np.random.normal(0, scale, (1, self.nodes[2]))) + np.random.normal(0, scale, (1, self.nodes[2])) #basis increased by x% and a flat y, x and y are seperate normal distribution random numbers

		self.new_L34 = basis_L34 * (1 + np.random.normal(0, scale, (self.nodes[2], self.nodes[3]))) + np.random.normal(0, scale, ((self.nodes[2], self.nodes[3]))) #basis increased by x% and a flat y, x and y are seperate normal distribution random numbers
		self.new_LB4 = basis_LB4 * (1 + np.random.normal(0, scale, (1, self.nodes[3]))) + np.random.normal(0, scale, (1, self.nodes[3])) #basis increased by x% and a flat y, x and y are seperate normal distribution random numbers

	def birth(self):
		self.L12 = self.new_L12
		self.LB2 = self.new_LB2

		self.L23 = self.new_L23
		self.LB3 = self.new_LB3

		self.L34 = self.new_L34
		self.LB4 = self.new_LB4


	def calc(self, parent1, parent2): #forward propogation
		next_two_obstacles = [parent2.Obstacles[parent2.Obstacles[:, 1] + parent2.Obstacles[:, 2] > parent1.Player_X_Pos][0], parent2.Obstacles[parent2.Obstacles[:, 1] + parent2.Obstacles[:, 2] > parent1.Player_X_Pos][1]]
		
		#prev.layer dot product weight layer + bias
		self.L1 = np.array([[-parent1.Player_Y_Pos+parent2.Ground_Y_Value, next_two_obstacles[0][1] - parent1.Player_X_Pos, screen.ScrollSpeed*10, next_two_obstacles[0][2], next_two_obstacles[0][3]]])
		self.L2 = np.maximum(self.L1 @ self.L12 + self.LB2, 0.01 * self.L1 @ self.L12 + self.LB2)
		self.L3 = np.maximum(self.L2 @ self.L23 + self.LB3, 0.01 * self.L2 @ self.L23 + self.LB3)
		self.L4 = np.maximum(self.L3 @ self.L34 + self.LB4, 0.01 * self.L3 @ self.L34 + self.LB4)
		
		#return self.L4
		return np.argmax((self.L4[:, 0], self.L4[:, 1], self.L4[:, 2]))

	def create_visual(self, parent2):
		global visualiser_arr
		global visualiser_weights_arr

		for obj in visualiser_arr:
			parent2.canvas.delete(obj)

		most_nodes = np.amax(self.nodes)

		circle_diameter = 20
		spacing_l = 50
		spacing_s = 10

		bbx1, bby1, bbx2, bby2 = [parent2.Canvas_Width - (len(self.nodes) * circle_diameter + (len(self.nodes) - 1) * spacing_l + spacing_s), spacing_s, parent2.Canvas_Width - spacing_s, circle_diameter * most_nodes + spacing_s * most_nodes] #x1, y1 is top left corner #bounding box

		visualiser_arr = np.array([[parent2.canvas.create_rectangle((bbx1 - spacing_s, bby1 - spacing_s), (bbx2 + spacing_s, bby2 + spacing_s), fill = "white", outline = "black"), -1, -1]]) #columns: tkinter object
		visualiser_weights_arr = np.array([[-1, -1, -1, -1]]) #tkinter object, first's level, first node, second node

		for _level in range(len(self.nodes)):
			for _node in range(self.nodes[_level]):
				visualiser_arr = np.append(visualiser_arr, [[parent2.canvas.create_oval(bbx1 + _level * (spacing_l + circle_diameter), (bby2 - bby1)/2 + bby1 - self.nodes[_level]/2 * circle_diameter - (self.nodes[_level] - 1)/2 * spacing_s + _node * (spacing_s + circle_diameter), bbx1 + circle_diameter + _level * (spacing_l + circle_diameter), (bby2 - bby1)/2 + bby1 - self.nodes[_level]/2 * circle_diameter - (self.nodes[_level] - 1)/2 * spacing_s + circle_diameter + _node * (spacing_s + circle_diameter)), _level, _node]], axis = 0)
				if _level < len(self.nodes) - 1:
					for _node2 in range(self.nodes[_level + 1]):
						visualiser_weights_arr = np.append(visualiser_weights_arr, [[parent2.canvas.create_line(bbx1 + circle_diameter + _level * (spacing_l + circle_diameter), (bby2 - bby1)/2 + bby1 - ((self.nodes[_level] - 1) * spacing_s + (self.nodes[_level] - 2) * circle_diameter + circle_diameter)/2 + (circle_diameter + spacing_s) * _node, bbx1 + (_level + 1) * (spacing_l + circle_diameter), (bby2 - bby1)/2 + bby1 - ((self.nodes[_level + 1] - 1) * spacing_s + (self.nodes[_level + 1] - 2) * circle_diameter + circle_diameter)/2 + (circle_diameter + spacing_s) * _node2), _level, _node, _node2]], axis = 0)

	def visualise(self, parent2):
		for _node in visualiser_arr[1::]:
			converter_node = {0: self.L1, 1: self.L2, 2: self.L3, 3: self.L4}
			lim = np.amax(np.abs(converter_node[_node[1]]))
			activation = converter_node[_node[1]][:, _node[2]][0] / lim

			converter_hex = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"A", 11:"B", 12:"C", 13:"D", 14:"E", 15:"F"}

			green = converter_hex[np.floor(15 * np.minimum(1, 1 + activation))] + converter_hex[np.floor((16 * np.minimum(1, 1 + activation) % 1) * 16)]
			red = converter_hex[np.floor(15 * np.minimum(1, 1 - activation))] + converter_hex[np.floor((16 * np.minimum(1, 1 - activation) % 1) * 16)]
			blue = converter_hex[np.floor(15 * np.minimum(1 + activation, 1 - activation))] + converter_hex[np.floor((16 * np.minimum(1 + activation, 1 - activation) % 1) * 16)]
			parent2.canvas.itemconfig(_node[0], fill="#" + red + green + blue)
		for _line in visualiser_weights_arr[1::]:
			converter_line = {0: self.L12, 1: self.L23, 2: self.L34}
			array = converter_line[_line[1]]
			lim = np.amax(np.abs(array))
			scaled_array = array/lim

			activation = scaled_array[_line[2], _line[3]]

			converter_hex = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"A", 11:"B", 12:"C", 13:"D", 14:"E", 15:"F"}

			red = converter_hex[np.floor(15 * np.minimum(1, 1 - activation))] + converter_hex[np.floor((16 * np.minimum(1, 1 - activation) % 1) * 16)]
			green = converter_hex[np.floor(15 * np.minimum(1, 1 + activation))] + converter_hex[np.floor((16 * np.minimum(1, 1 + activation) % 1) * 16)]
			blue = converter_hex[np.floor(15 * np.minimum(1 - activation, 1 + activation))] + converter_hex[np.floor((16 * np.minimum(1 - activation, 1 + activation) % 1) * 16)]

			parent2.canvas.itemconfig(_line[0], fill="#" + red + green + blue)

	def __str__(self):
		return str(self.L1) + "\n\n" + str(self.L12) + "\n\n" + str(self.LB2) + "\n\n\n" + str(self.L2) + "\n\n" + str(self.L23) + "\n\n" + str(self.LB3) + "\n\n\n" + str(self.L3) + "\n\n" + str(self.L34) + "\n\n" + str(self.LB4) + "\n\n\n" + str(self.L4)

#code
amount_of_players = 250
trial = 1
generation = 1

def initiate():
	global screen
	global players
	global neural_networks

	screen = game()

	screen.remove_out_of_bounds()

	players = np.array([])
	neural_networks = np.array([])

	for index in range(amount_of_players):
		players = np.append(players, player(screen, index))
		neural_networks = np.append(neural_networks, neural_network(players[index], screen))

def end(event):
	print("best_results:", sorted_neural_networks[0, 0])
	screen.root.destroy()
	quit()

initiate()

screen.root.bind("<space>", end)

last_average = 0

last_surviving = 0

max_time = 20

initial_time = datetime.now()

surviving_players = 0

visualiser_arr = np.array([])

visualiser_weights_arr = np.array([])

while True: #change later to allow for multiple 'games'
	#gaming
	screen.refresh()
	for index in screen.alive_players:
		decision = neural_networks[index].calc(players[index], screen)
		if decision == 0:
			pass
		elif decision == 1:
			players[index].jump()
		elif decision == 2:
			players[index].drop()

		players[index].move(screen)

		players[index].gravity()

		players[index].collision_detection(screen)

	if (datetime.now() - initial_time).total_seconds() > max_time:
		for index in screen.alive_players:
			players[index].Dead = True
					
			players[index].Score += (screen.distance + players[index].Player_X_Pos) * 2
			surviving_players += 1
			max_time += 1

			screen.canvas.delete(players[index].player)
			screen.alive_players = screen.alive_players[screen.alive_players != index]
			screen.alive_neural_networks = screen.alive_neural_networks[screen.alive_neural_networks != neural_networks[index]]

	#new trial/gen
	if len(screen.alive_players) == 0:
		if trial == 3:
			#create children
			scores = np.array([])

			for _player in players:
				scores = np.append(scores, _player.Score)
				_player.Score = 0

			print("generation:", generation)

			if surviving_players == 0:
				print("\naverage score:", np.average(scores)/3)
				if last_average == 0:
					print("improvement: N/A")
				else:
					print("improvement:", str(np.round(((np.average(scores)/3) / last_average - 1) * 100, decimals = 2)) + "%")
				print("\nhighest score:", np.max(scores)/3)
			else:
				print("\nlife span:", max_time - surviving_players)
				print("\nsurviving_players:", surviving_players)
				if last_surviving == 0:
					print("improvement: N/A")
				else:
					print("improvement:", str(np.round((surviving_players / last_surviving - 1) * 100, decimals = 2)) + "%")
				
			print("\n==========================\n")

			last_average = np.average(scores)/3
			last_surviving = surviving_players

			scores_sorted_indexes = np.argsort(scores)

			sorted_neural_networks = np.array([neural_networks[scores_sorted_indexes[::1]], scores[scores_sorted_indexes[::1]]]).T

			fit_parents = sorted_neural_networks[int(np.ceil(amount_of_players / 5 * 4)):]
			unfit_parents = sorted_neural_networks[:int(np.ceil(amount_of_players / 5 * 4))]

			#reproduce
			for neural_network, score in unfit_parents:
				parents = random.choices(fit_parents, k=2)

				neural_network.reproduce(parents[0][0], parents[0][0]) #make the two reproduce
				neural_network.birth() 


			trial = 0
			generation += 1
			surviving_players = 0

		#destroy and recreate
		screen.recreate()

		screen.remove_out_of_bounds()

		for _player in players:
			_player.respawn(screen)

		if generation > 1:
			sorted_neural_networks[:, 0][np.isin(sorted_neural_networks[:, 0], screen.alive_neural_networks)][0].create_visual(screen)
			sorted_neural_networks[:, 0][np.isin(sorted_neural_networks[:, 0], screen.alive_neural_networks)][0].visualise(screen)

		trial += 1
		initial_time = datetime.now()

	if generation > 1:
		sorted_neural_networks[:, 0][np.isin(sorted_neural_networks[:, 0], screen.alive_neural_networks)][0].visualise(screen)
