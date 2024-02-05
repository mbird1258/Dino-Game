Dino Game with ML from Scratch
==============================
My goal for this project was to ensure that I can implement a basic neural network and make it change to complete a task, which I decided to achieve by making a game with simple rules that should be possible for machine learning to complete with ease. In addition, I should be able to mostly reuse the neural network class for any future projects. 

Attempt 1 (Random Changes)
--------------------------

Attempt 2 (Directed Changes)
----------------------------

### Plan
My plan for this game is to replicate the iconic dinosaur game and train a model that can achieve the highest possible score in my version of the game. 

### Execution

#### The Game

##### The Screen
The program uses tkinter to create the visuals for the screen and obstacles. The coordinates of each obstacle is added to a 2d matrix for use in collision detection and to delete off screen objects. 

To scroll the screen, the program simply changes every obstacle’s x value by the scroll speed in the left direction. When an obstacle goes off screen, the program simply deletes the obstacle and then recreates one so that the obstacles never run out. 

##### The Player
The player is simply a rectangle with three possible actions, jumping, dropping and ducking. Jumping gives it a positive y velocity that is reduced every frame. Dropping sets the player’s y velocity to a set negative number to make it fall faster. Ducking allows the player to reduce its hitbox while on the ground. 

#### Neural Network

##### Structure
The neural network's structure can be changed based on the arguments it takes in, but the base parameters take in the player height above ground, next obstacle distance, screen scroll speed, and the next obstacle upper y value. Then, it passes through one hidden layer with 24 nodes each, ending up in an output layer with 4 nodes. The player decides to jump, drop, duck or do nothing based on the output node with the highest value. 

##### Activation Function

##### Instance Normalization

##### Q learning and Backpropagation

##### Visualization

##### Hyperparameters

### The Journey
