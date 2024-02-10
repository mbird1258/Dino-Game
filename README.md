Dino Game with ML from Scratch
==============================
My goal for this project was to ensure that I can implement a basic neural network and make it change to complete a task, which I decided to achieve by making a game with simple rules that should be possible for machine learning to complete with ease. In addition, I should be able to mostly reuse the neural network class for any future projects. 

Attempt 1 (Random Changes)
--------------------------

### Plan
My plan for this game is to replicate the iconic dinosaur game and train a model that can achieve the highest possible score in my version of the game. 

### Execution

#### The Game

##### The Screen
The program uses tkinter to create the visuals for the screen and obstacles. The coordinates of each obstacle are added to a 2D matrix for use in collision detection and to delete off screen objects. 

To scroll the screen, the program deducts the scroll speed from every obstacle’s x value. When an obstacle goes off screen (right x value < 0), the program deletes the obstacle and then recreates another one in its place so that the obstacles never run out. 

##### The Player
The player is a rectangle with two possible actions: jumping and dropping. Jumping gives it a positive y velocity that is reduced every frame. Dropping sets the player’s y velocity to a set negative number to make it fall faster. If its rectangular hitbox collides with any of the obstacles, it dies. 

There are 250 players that play the game at once. 

#### Neural Network

##### Structure
The neural network is a feed forward neural network that takes in the player’s height above ground, next obstacle’s distance, screen scroll speed, the next obstacle's width, and the next obstacle's height. Then, it passes through two hidden layer with 5 nodes each, ending up in an output layer with 3 nodes. The player decides to jump, drop, or do nothing based on the output node with the highest value. 

#### Forward Propagation
For forward propagation, the neural network follows the following process: 
1. Dot product between inputs and the weight layer + bias. 
2. Feed the outputs through Leaky Relu activation function.
3. Repeat for following layer with outputs as new inputs. 

#### 'Learning'
After each generation of 3 trials, the top 20% of models experience no changes. 79% of the remaining models are completely redone, taking 2 models from the top 20%, taking the average weights, and making slight changes to them. The last 1% is a completely randomly created model. The idea is that the top 20% of models can only get better, the 79% will eventually find better solutions and enter the top 20%, and the last 1% will eventually lead to finding the best neural network given infinite time. 

#### Visualization

##### Tkinter object positioning:

<img src="https://github.com/mbird1258/Dino-game-with-machine-learning-from-scratch/assets/132913097/c54b3f07-b2b1-47f5-bad9-abc316e58ea3" width="387" height="500">

##### Tkinter object coloring:

For each node layer, get the absolute values and take the maximum. Then, divide each node's value in the layer by this maximum so that the values lie between -1 and 1. Then, we can feed these values into the functions defined [here](https://www.desmos.com/calculator/bzltel0e6v) to get the rgb values for each node. 

For each weight, we do a similar thing, taking the maximum absolute value of the weights in the layer, then divide each weight's value by the maximum, and then feed the values into the same functions to get the corresponding rgb values for the weight lines. 

### Results

This method worked surprisingly well, probably because of the simplicity of the problem (unlike attempt 2, we just need to find the best move, not the exact rewards for each move in each state). Overall, I think this was a great success, but the learning algorithm was about as simple as it can get and isn't very applicable to other problems. In addition to this, I think that taking the average weights between 2 models makes no sense, as taking the average can lead to vastly different values in node layers, while it makes more sense to just take 1 model and make small changes to it until it converges on a solution. 

Attempt 2 (Directed Changes)
----------------------------

### Plan
My plan for this game is to replicate the iconic dinosaur game and train a model that can achieve the highest possible score in my version of the game. 

### Execution

#### The Game

##### The Screen
The program uses tkinter to create the visuals for the screen and obstacles. The coordinates of each obstacle are added to a 2D matrix for use in collision detection and to delete off screen objects. 

To scroll the screen, the program deducts the scroll speed from every obstacle’s x value. When an obstacle goes off screen (right x value < 0), the program deletes the obstacle and then recreates another one in its place so that the obstacles never run out. 

##### The Player
The player is a rectangle with three possible actions: jumping, dropping and ducking. Jumping gives it a positive y velocity that is reduced every frame. Dropping sets the player’s y velocity to a set negative number to make it fall faster. Ducking allows the player to reduce its hitbox while on the ground. If its rectangular hitbox collides with any of the obstacles, it dies. 

#### Neural Network

##### Structure
The neural network is a feed forward neural network that takes in the player’s height above ground, next obstacle’s distance, screen scroll speed, and the next obstacle upper y value. Then, it passes through one hidden layer with 24 nodes, ending up in an output layer with 4 nodes. The player decides to jump, drop, duck, or do nothing based on the output node with the highest value. 

I chose 24 hidden nodes to make sure that the neural network could accurately represent the return of each action(which it is learning to do because it learns based off of Deep Q learning). [(Desmos link)](https://www.desmos.com/calculator/5pipewvgfe)

#### Forward Propagation
For forward propagation, the neural network follows the following process: 
1. Dot product between inputs and the weight layer + bias. 
  1. Bias is added through adding a 1 to the start of the inputs vector and adding in a row to the top of the weights matrix, which represents the bias for each output node. 
2. Feed the outputs through the activation function.
3. Instance normalize the outputs. [(Desmos link)](https://www.desmos.com/calculator/8lrwjurthd)
4. Repeat for following layer with outputs as new inputs. 

#### Activation Function
There are many activation functions to choose from. [(Desmos link)](https://www.desmos.com/calculator/zehhcy84z0) Each function takes in one input and returns one output. Each function also has a derivative for use in backpropagation. The list of activation functions is as follows:
1. relu
2. lrelu
3. softplus
4. sigmoid
5. tanh
6. swish
7. mish
8. RBF
9. RBFx
10. ELU

#### Backpropagation
[Guide that I used](https://www.3blue1brown.com/topics/neural-networks)

Backpropagation is a method of taking a neural network that produces bad results and changing it over many iterations such that it produces good results. It does this through using principles of Calculus to minimize a function that measures how bad a given neural network is, called a loss function. The most common loss function used is called mean squared error, in which the function takes the difference between what the function outputted and what it should have outputted (that we tell it), and then takes the mean of the function. The function represented mathematically is as follows: $C = mean((a-y)^2)$, where $a$ represents the outputs and $y$ represents what the function should have returned. 

Before minimizing the cost function, it is important to understand what values we are trying to change in order to make our neural network better. The neural network cannot change what inputs/state it takes in, but rather only the weights and biases it uses to take the inputs and produce an output from them. 

Because $a$ itself is a function of the inputs of the neural network, we can rewrite the cost function as a long function that takes in the inputs and outputs the output and take the derivative. However, it is much easier to break this down into steps through the chain rule. To minimize the cost function, we can take the partial derivative of $C$ with respect to $a$, which leaves us with the function $2(a-y)$(Note that if the loss function is rewritten as $C = mean((y-a)^2)$, the derivative would be $-2(y-a)$ or still $2(a-y)$ due to the chain rule). Then, we can take the partial derivative of $C$ with respect to the previous layer, called $a$, then repeat from $a$ to $a$ before it was put through the activation function, called $z$, and then from $z$ to $IN(a')$, where $IN(a')$ represents the instance normalized values of $a'$. We can repeat this process all the way until we get down to the inputs. However, we have found how we want the nodes on each layer to change, but we cannot change these directly. Instead, we have to find the derivative of $z$ with respect to $w'$ and $b'$ for all $z$, $w$, and $b$ in the neural network. Note that I am using each $'$ to represent a layer from the last layer as opposed to taking the derivative. A nice way to visualize this is with a tree:

<img src="https://github.com/mbird1258/Dino-game-with-machine-learning-from-scratch/assets/132913097/e070c99d-04d7-4e84-af29-4020a1fdc5fd" width="387" height="500">

[Site used to get dot product partial derivatives](https://cs231n.github.io/optimization-2/)

[Site used to get instance normalization derivatives](https://en.wikipedia.org/wiki/Batch_normalization#Backpropagation)

It is important to note that each time we calculate a derivative for $a$ on any layer, we should clip the value to be below a certain threshold to ensure we don't experience exploding gradients, as is further explained [here](https://neptune.ai/blog/vanishing-and-exploding-gradients-debugging-monitoring-fixing). 

To get to the final derivatives, we can use the chain rule to multiply each derivative by the next down the path of the tree to get to the weight layer that we want to change. In the tree, bias was not shown, but since we use the first layer of the weight matrix to represent bias and add a 1 into the node layer to work together with this, we can calculate the partial derivative of bias along with the weights and it returns what we would expect. 

The final operation to perform is to adjust the weight matrices in accordance to our derivatives. If our derivative for a value is positive, meaning that increasing that weight's value increases the cost of the neural network (in other words, makes it worse), we want to reduce that weight. In other words, we want to subtract each weight's derivative from said weights to make our model better. 

#### Deep Q Learning
[Guide that I used](https://www.youtube.com/watch?v=rFwQDDbYTm4)

Deep Q Learning is a method of backpropagation for reinforcement learning problems, such as the dinosaur game. The first thing to note is that the neural network created by the Deep Q Learning algorithm represents the expected returns of each action as opposed to learning to choose the most rewarding action. While these both end up performing the same action when we act based on the highest value, they do not necessarily represent the same things, and representing expected returns usually takes a combination of many nodes. [(Desmos link)](https://www.desmos.com/calculator/5pipewvgfe)

The big issue with reinforcement learning problems that Deep Q Learning strives to solve is that we don't know how good each action is at the moment. For example, if we jump before an obstacle, we don't know if that jump was a good choice until countless frames later, when we either live or die. To solve this issue, Q Learning uses the immediate reward of an action, plus its own estimate of the future reward multiplied by some discount factor, as the goal/label for backpropagation. In the case of a terminating (final) state, there are no future rewards, and thus the goal/label is simply the immediate reward. 

This may seem counterintuitive, as we are trying to improve our bad neural network results through the neural network's own predictions. However, the key to this algorithm is that in the terminating state, we know the exact value our goal/label for backpropagation should have. From this, we can learn the return at the terminating state, then learn the return at the second last state using our good prediction of the next (terminating) state's return. This continues on all the way until the neural network learns to deal with all states. 

As for the discount factor, if we choose 1 for the discount factor, meaning that we add immediate reward with all future rewards, our numbers will spiral out of control in the starting state, approaching infinity based on how long the episode (in the case of dino game, one life) lasts. This is a big issue, especially in the case of the dinosaur game, as if we learn how to estimate return better, we will live longer and then have to relearn our values, as if we live longer, the episode lasts longer, and our initial returns will be higher. If we choose 0 for the discount factor, we lose all data of the future, which defeats the entire purpose of the Q Learning algorithm. For example, if we jump, we only get rewarded many frames after we pass the obstacle, but with 0 as our discount factor, we would not reward jumping if we live, nor punish jumping if we die. Using numbers close to 1 as a discount factor falls into the same issue as using 1 for the discount factor and increases learning time due to the label being $r + γ * max(Q(s'))$, so when γ is high, little change is made. On the other hand, using numbers close to 0 can make a neural network shortsighted(ex. dino jumping even though it will die right after). 

The formula for Deep Q Learning that I used is:
  $Q'(s, a) = r + γ * max(Q(s'))$

s is the current state/inputs of the neural network, a is the action taken (not always the highest value if exploring), and s' is the next state. 

Q'(s, a) is the better estimate for the neural network that we are using as a goal/label for backpropagation, r represents the immediate reward, γ represents the discount factor, and max(Q(s')) is the current model's estimate for the highest return in the next state. 

Another core aspect of Deep Q Learning is the exploration rate. If we always only take the best action, the neural network may never learn about how good another action might be because it will never take it. For example, if in a given state, we attribute a value of -100 to action 1, we will never ever take this action, and thus we won't be able to use Deep Q Learning to get the goal to improve our prediction of the return of action 1, which might in fact be the best action. In other words, we need a random chance of taking random choices so that we can perfect all actions' returns. This is especially needed at the start, and as we get further down the line, it often proves helpful to reduce the exploration rate so that we can perfect our predictions of the top actions taken that matter most. 

The last core aspect of Deep Q Learning is replay memory. Essentially, the program stores all the information the neural network requires to perform Deep Q Learning and then backpropagation, in a large matrix, in which each set of information(called a transition) is a row. When we're performing Deep Q Learning and backpropagation, we randomly sample 64 (or some other number) of these transitions to perform backpropagation on to improve our dino's neural network. This is important as it both allows us to train on each transition multiple times and eliminates the issue of the neural network training on many similar states in a row and becoming biased towards one action that only fits those states. For example, if we have our model and are not using replay memory, we might start off with no obstacle in sight, and the most rewarding action would simply be to do nothing. Over many frames, we learn that doing nothing is the perfect action to take and become biased towards it. But suddenly, an obstacle appears, and we die. However, we have only learned that doing nothing was bad at the end of the episode, and after training and running the next round, we will become biased towards doing nothing again. 

Replay memory requires three parameters: minimum replay memory size, maximum replay memory size, and batch size. Minimum replay memory size requires a set amount of transitions before the model starts training, ensuring that the problems replay memory is trying to solve doesn't affect the first training rounds. The maximum replay memory size replaces the oldest transitions with newer transitions once the transitions exceed a certain amount to save memory. Batch size determines how many transitions to sample from replay memory each training round, with higher numbers meaning longer time spent training due to more training data but also more improvement in the model. 

#### Hyperparameters

The hyperparameters and their functions are listed below:
- Save interval
  - How often to save neural network weights to computer
- Nodes
  - How many nodes in each layer, including first, hidden, and last layers
- Adjustment rate
  - How big the step size per round of backpropagation
- Adjustment rate falloff
  - The value multiplied to adjustment rate per episode to reduce step size to converge on an answer
- Loss function
  - Which loss function to use
- Activation functions
  - Which activation function to use each node layer
- Gamma
  - Discount factor
- Gamma increase
  - How fast gamma increases until limit
- Gamma max
  - Max value for gamma
- Minibatch minimum
  - Minimum replay memory size before training starts
- Minibatch maximum
  - Maximum replay memory size before oldest transitions are replaced
- Weight decay factor
  - If using loss function that factors in weight decay, the closer to 1, the more it penalizes high weights, and the closer to 0, the less it penalizes high weights.
- Exploration rate
  - How much the neural network initially chooses random actions
- Exploration rate falloff
  - How fast the exploration rate reduces over time
- Minimum exploration rate
  - Minimum exploration rate to ensure training doesn't completely stop

#### Visualization

##### Tkinter object positioning:

<img src="https://github.com/mbird1258/Dino-game-with-machine-learning-from-scratch/assets/132913097/c54b3f07-b2b1-47f5-bad9-abc316e58ea3" width="387" height="500">

##### Tkinter object coloring:

For each node layer, get the absolute values and take the maximum. Then, divide each node's value in the layer by this maximum so that the values lie between -1 and 1. Then, we can feed these values into the functions defined [here](https://www.desmos.com/calculator/bzltel0e6v) to get the rgb values for each node. 

For each weight, we do a similar thing, taking the maximum absolute value of the weights in the layer, then divide each weight's value by the maximum, and then feed the values into the same functions to get the corresponding rgb values for the weight lines. 

#### Saving the Results

To save the neural network, we save the numpy array to a predetermined file with np.save every 50 episodes. To get the neural network data back, we use np.load to set the weight layers equal to the ones saved in the same file. 

### Results

The neural network performs pretty well, consistently learning to jump over the cacti. However, it struggles to learn to drop after passing the cacti and thus often gets stuck not jumping in time due to a lack of space between cacti. This can likely be fixed through training for longer or perfecting the hyperparameters of the neural network, but I'm satisfied with these results as my version of the dinosaur game is, at least from what I can tell, considerably harder than the real dinosaur game. 
