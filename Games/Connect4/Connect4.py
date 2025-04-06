class game_connect4:
    #Class handling game mechanics
    
    def __init__(self):
        self.state = [[0 for x in range(7)] for y in range(7)]
        
    def __getitem__(self, start):
        return(self.state[start])
    
    def over(self):
        #Checks if game is over - there are four tokens horizontally, diagonally or vertically

        #TODO save winning configuration and output it as var
        
        chck1 = ('1, ')*3+'1'
        chck2 = ('2, ')*3+'2'
        #Check horizontaly
        if any(chck1 in str(x) for x in g.state) or any(chck2 in str(x) for x in g.state):
            return True
        else:
            #Check verticaly
            for j in range(7):
                temp = [self.state[i][j] for i in range(7)]
                if chck1 in str(temp) or chck2 in str(temp):
                    return True
            else:
                #Check diagonals
                for i in range(0,4):
                    temp = [self.state[i+j][j] for j in range(0,7) if (i+j<=6)]
                    if chck1 in str(temp) or chck2 in str(temp):
                        return True
                    temp = [self.state[j][i+j] for j in range(0,7) if (i+j<=6)]
                    if chck1 in str(temp) or chck2 in str(temp):
                        return True
                    temp = [self.state[i+j][6-j] for j in range(7) if (i+j<=6)]
                    if chck1 in str(temp) or chck2 in str(temp):
                        return True
                    temp = [self.state[6-j][i+j] for j in range(7) if (i+j<=6)]
                    if chck1 in str(temp) or chck2 in str(temp):
                        return True
                return(False)
            
    def valid(self,column):
        #Checks if column has any free spaces
        if column>6:
            return(False)
        return(any([i[column]==0 for i in self.state]))
        
    def AddToken(self,column,token):
        #Adds token to selected column
        if self.valid(column):
            for i in range(6,-1,-1):
                if self.state[i][column] ==0:
                    self.state[i][column] = token
                    return(i,column)


# Function to handle different non-human agents
def handle_agent_turn(player, g, SCREEN, token, com, font, objects):
    if player_types[player] == "random":
        # not really needed, made pause for better game feel
        pygame.time.delay(500) 
        col = get_player_move(player)
        if col is not None and g.valid(col):
            row, column = g.AddToken(col, player)
            new_token = AnimateToken(SCREEN, token, column, row, com, font, objects)
            objects.insert(0, new_token)
            return row, column
    return None, None

        
# Rendering and image manipulation
def LoRs(source,scale):
    #Loads and Resizes the image
    img = pygame.image.load(source)
    width = img.get_rect().width
    height = img.get_rect().height
    res_img = pygame.transform.scale(img, (width//2, height//2))
    return(res_img)


def Display_objects(screen,objects):
    # Draw all objects in the list
    for thing,crd in objects:
        screen.blit(thing,crd)

        
def Display_instructions(screen,text,font,over=False):
    # This function displays instruction or announces win
    pygame.draw.rect(screen,(150,150,150),(0,600,600,50))
    screen.blit(font.render(text, False, (255,0,0)), (10, 600))
    

def AnimateToken(screen,token,column,row,com,font,objects):
    # Make animation of token falling into the board
    BLACK = (0, 0, 0)
    x0 = 75 +(column)*64.2
    y0 = 458 - 506.88
    y1 = 458 - (6-row)*63.36
    while y0 <= y1:
        y0 += 6.336
        
        screen.fill(BLACK)
        Display_instructions(screen,com,font)
        Display_objects(screen,objects)
        screen.blit(token, (x0,y0))
        screen.blit(board, (60,60))
        pygame.display.update()
        clock = pygame.time.Clock()
        clock.tick(600)
        
    #Return token to be appened to the list of objects
    new_token = (token,(x0,y0,))
    return(new_token)


def get_player_move(player):
    if player_types[player] == "human":
        return None  # handled through keypress
    elif player_types[player] == "random":
        valid_moves = [c for c in range(7) if g.valid(c)]
        return random.choice(valid_moves) if valid_moves else None


#Initialize pygame
import numpy as np
import pygame, random
from time import sleep

player_types = {1: "human", 2: "random"}  # or "human", "random"


pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Arial', 30)

# Window size
WIDTH = 600
HEIGHT = 650

# Define display object
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Connect 4')

# Colors palette
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paths to images
board_path = r"board_t.png"
token1_path = r"token1.png"
token2_path = r"token2.png"

# Load and scale images
board = LoRs(board_path,2)
token1 = LoRs(token1_path,2)
token2 = LoRs(token2_path,2)

# Initialize game
g = game_connect4()

objects = [(board,(60,60))]

#Keys which can be used by player
controls = [pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,
            pygame.K_5,pygame.K_6,pygame.K_7]

player = 1
com = "It's player's 1 turn."
token = token1
gquit = False

import pygame
from time import sleep
import random

# Wrap the whole game logic inside this loop
restart = True
while restart:
    restart = False  # Default is not restarting
    g = game_connect4()
    objects = [(board,(60,60))]
    player = 1
    com = "It's player's 1 turn."
    token = token1
    gquit = False

    while not g.over() and not gquit:
        for event in pygame.event.get():
            # Press ESC to quit anytime
            if event.type == pygame.QUIT:
                gquit = True
                
            # if any key is pressed
            elif event.type == pygame.KEYDOWN:
                if player_types[player] == "human":
                    if event.key == pygame.K_ESCAPE:
                        gquit = True
                    # check if player is pressing column number or some random keys
                    elif event.key in controls:
                        col = int(event.key - 48)
                        if g.valid(col - 1):
                            row, column = g.AddToken(col - 1, player)
                            new_token = AnimateToken(SCREEN, token, column, row, com, font, objects)
                            objects.insert(0, new_token)
                            if not g.over():
                                player = 2 if player == 1 else 1
                                com = f"It's player's {player} turn."
                                token = token2 if player == 2 else token1

        # Agent turn
        if player_types[player] != "human" and not g.over():
            row, column = handle_agent_turn(player, g, SCREEN, token, com, font, objects)
            if row is not None:
                if not g.over():
                    player = 2 if player == 1 else 1
                    com = f"It's player's {player} turn."
                    token = token2 if player == 2 else token1


        if not gquit:
            SCREEN.fill(BLACK)
            Display_instructions(SCREEN, com, font)
            Display_objects(SCREEN, objects)
            pygame.display.flip()

    if g.over():
        winner = 2 if player == 2 else 1
        com = f"Player {winner} won!"
        SCREEN.fill(BLACK)
        Display_instructions(SCREEN, com + " Press Enter to restart.", font)
        Display_objects(SCREEN, objects)
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    gquit = True
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        gquit = True
                        waiting = False
                    elif event.key == pygame.K_RETURN:
                        restart = True
                        waiting = False

pygame.quit()
