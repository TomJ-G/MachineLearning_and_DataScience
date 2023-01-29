#Game playable for AI
#Loading libraries
import pygame
import random
from time import sleep
import numpy as np
from collections import deque
import torch 
import random 

#============== MODEL and AGENT ==================#

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0005

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11,512,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 750 - self.n_game
        final_move = [0,0,0]
        if(random.randint(0,200)<self.epsilon):
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0) # prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move
    

#Linear Q-net
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
 
    def save(self, file_name='model_name.pth'):
        model_folder_path = r'C:\Users\galic\Documents\Python Scripts\Snake game'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
class QTrainer:
    def __init__(self,model,lr,gamma):
        #Learning Rate for Optimizer
        self.lr = lr
        #Discount Rate
        self.gamma = gamma
        #Linear NN defined above.
        self.model = model
        #optimizer for weight and biases updation
        self.optimer = optim.Adam(model.parameters(),lr = self.lr)
        #Mean Squared error loss function
        self.criterion = nn.MSELoss()


    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)

        # if only one parameter to train , then convert to tuple of shape (1, x)
        if(len(state.shape) == 1):
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )

        # 1. Predicted Q value with current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        self.optimer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward() # backward propagation of loss
        self.optimer.step()

def train_short_memory(self, state, action, reward, next_state, done):
    self.trainer.train_step(state, action, reward, next_state, done)

def train_long_memory(self):
    if (len(self.memory) > BATCH_SIZE):
        mini_sample = random.sample(self.memory, BATCH_SIZE)
    else:
        mini_sample = self.memory
    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.trainer.train_step(states, actions, rewards, next_states, dones)



#============== Functions ==============#
def draw_borders(s, x, y, w, h, bw, c):
    pygame.draw.rect(s, c, (x, y, w, bw))
    pygame.draw.rect(s, c, (x, y+h-bw, w, bw))
    pygame.draw.rect(s, c, (x, y, bw, h))
    pygame.draw.rect(s, c, (x+w-bw, y, bw, h))


def draw_snake(body):
    for x,y in body:
        pygame.draw.rect(screen, snake_color, [x, y, snake_block, snake_block])

    
def move_snake(body,x_velocity,y_velocity):
    x,y = body[0]
    #x +=
    body.pop(-1)
    body.insert(0,(x+x_velocity,y+y_velocity))

#CHANGE THIS to reflect (1,0,0) actions   
def get_direction(action,cur):
    if action[0] == 1:
        return(directions[cur],cur)
    elif action[1] == 1:
        return(directions[(cur+1)%4],(cur+1)%4)
    else:
        #action == 2
        return(directions[(cur-1)%4],(cur-1)%4)

def collision_wall(x,y):
    if x > p_size[0] or x <= 0 or y >= p_size[1] or y <= 0:
        return(True)
    else:
        return(False)

def danger(cur,x,y):
    #Wall check
    #Danger in straight direction means x,y + direction
    wall_straight = collision_wall(x+directions[cur][0],y+directions[cur][1])
    
    #For left and right I need to use get direction func
    dl,cl = get_direction((0,0,1),cur)
    wall_left     = collision_wall(x+dl[0],y+dl[1])
    
    dr,cr = get_direction((0,1,0),cur)
    wall_right    = collision_wall(x+dr[0],y+dr[1])
    
    #Body check
    body_straight = (x+directions[cur][0],y+directions[cur][1]) in snake_body[1:]
    body_left     = (x+dl[0],y+dl[1]) in snake_body[1:]
    body_right    = (x+dr[0],y+dr[1]) in snake_body[1:]
    
    danger_straight = int(wall_straight|body_straight)
    danger_left     = int(wall_left|body_left)
    danger_right    = int(wall_right|body_right)
    
    return(danger_straight,danger_left,danger_right)

def get_state(x,y,cur,fx,fy):
    """
    x,y     - snake head position
    cur - velocity (direction)
    fx,fy   - food position
    """
    x_v,y_v = directions[cur]
    #Directions are up, down, left, right
    direction = (int(y_v==-10),int(x_v==10),int(y_v==10),int(x_v==-10))
    #Danger is checked as collision with food or wall
    d = danger(cur,x,y)
    #Is foot up, down, left, right
    food   = (int(fy<y),int(fx>x),int(fy>y),int(fx<x))
    return( np.array(d+direction+food,dtype=int) )   
    

#=======================================#
plot_scores = []
plot_mean_scores = []
reward = 0
# == Define the game == #
    
# Initialize pygame
pygame.init()

# Set the screen size
SX, SY = 340, 350
size = (SX, SY)
screen = pygame.display.set_mode(size)

#Set the play area
PX,PY = (320,240)
p_size = (PX,PY)


# Set the title of the window
pygame.display.set_caption("Snake Game")


# Set the colors
snake_color = (169, 27, 96)#A91B60
eaten_color = (255, 0, 128)#FF0080
food_color = (0, 255, 0)
back_color = (0,0,0)
border_color = (255,255,255)


# Initialize the snake
x1 = 100
y1 = 100
snake_block = 10
snake_speed = 10
snake_body = [(x1,y1),(x1,y1+10),(x1,y1+20)]


# Set the initial position of the food 
#and make sure that food won't spawn at the same position as snake
foodx = random.randint(1,(SX/10)-1)*10
foody = random.randint(1,(PY/10)-1)*10
while foodx == x1 and foody ==y1:
    foodx = random.randint(1,(SX/10)-1)*10
    foody = random.randint(1,(PY/10)-1)*10
    
    
# Set the initial player score
score = 0
record = 0
total_score = 0
agent = Agent()

#Set the text box
pygame.font.init() 
font = pygame.font.Font('freesansbold.ttf', 32)
text = font.render('Score: '+str(score), True, (0,0,0), (100,100,100))
textRect = text.get_rect()
textRect.center = (100, 340)

eaten = []
directions = ((0,-10),(10,0),(0,10),(-10,0))
cur = 0
x1_velocity, y1_velocity = directions[cur]

iteration = 0

game = True

# Run the game loop

action = (1,0,0)

while game==True:
    done = False
    state_old = get_state(x1,y1,cur,foodx,foody)
    action = agent.get_action(state_old)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game = False
            pygame.quit()
            quit()
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                game = False
                sleep(0.5)
                pygame.quit()
                quit()
                break
    
    iteration +=1           
    #state = get_state(x1,y1,cur,foodx,foody)
    #We need to determine direction based on action
    xy_vel, cur = get_direction(action,cur)
    x1_velocity, y1_velocity = xy_vel
                
    if game == True:
        x1,y1 = snake_body[0]
        neck = snake_body[1:]

        # Clear the screen
        screen.fill(back_color)

        #Draw text
        screen.blit(text, textRect)

        #Draw borders
        draw_borders(screen, 0, 0, SX, PY+10, 10, border_color)

        # Draw the snake
        draw_snake(snake_body)

        # Draw the food
        pygame.draw.rect(screen, food_color, [foodx, foody, snake_block, snake_block])

        # Process eaten food
        for e in eaten:
            if e in snake_body:
                pygame.draw.rect(screen, eaten_color, [e[0], e[1], snake_block, snake_block])
            else:
                snake_body.append(eaten.pop(0))

        # Update the screen
        pygame.display.update()

        # Set the frame rate of the game
        clock = pygame.time.Clock()
        clock.tick(snake_speed+70)
        reward = 0
        # Check for collision with the walls
        if collision_wall(x1,y1):
            done = True
            reward = -10
            text = font.render('Score: '+str(score), True, (0,0,0), (100,100,100))
            #Reset snake position, reset movement

        #Check for collision with body
        if (x1,y1) in neck:
            done = True
            reward = -10
            text = font.render('Score: '+str(score), True, (0,0,0), (100,100,100))
            #Reset snake position, reset movement

        # Check for collision with the food
        if x1 == foodx and y1 == foody:
            eaten.append((foodx,foody))
            reward = 10
            score += 1
            iteration = 0
            text = font.render('Score: '+str(score), True, (0,0,0), (100,100,100))
            
            foodx = random.randint(1,(PX/10)-1)*10
            foody = random.randint(1,(PY/10)-1)*10
            while (foodx,foody) in snake_body:
                foodx = random.randint(1,(PX/10)-1)*10
                foody = random.randint(1,(PY/10)-1)*10
            
        # Check if iterations are to big
        if iteration > 100*len(snake_body):
            reward = -10
            done = True
            #re-initialize game...

        move_snake(snake_body,x1_velocity,y1_velocity)
        
        state_new = get_state(x1,y1,cur,foodx,foody)
        # train short memory
        agent.train_short_memory(state_old,action,reward,state_new,done)
        #remember
        agent.remember(state_old,action,reward,state_new,done)

        if done:
            iteration = 0
            x1_velocity = 0
            y1_velocity = 0
            x1 = 100
            y1 = 100
            foodx = random.randint(1,(PX/10)-1)*10
            foody = random.randint(1,(PY/10)-1)*10
            snake_body = [(x1,y1),(x1+10,y1),(x1+20,y1)]
            cur = 0
            action = (1,0,0)
            
            # Train long memory,plot result
            #game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if(score > reward): # new High score 
                reward = score
                agent.model.save()
            print('Game:',agent.n_game,'Score:',score,'Record:',record)

            plot_scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            #plot(plot_scores,plot_mean_scores)
            score = 0