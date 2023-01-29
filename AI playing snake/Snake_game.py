#Game playable by user
#Loading libraries
import pygame
import random
from time import sleep
# == Define the game == #
    
# Initialize pygame
pygame.init()

# Set the screen size
SX, SY = 400, 400
size = (SX, SY)
screen = pygame.display.set_mode(size)

#Set the play area
PX,PY = (380,350)
p_size = (PX,PY)

# Set the title of the window
pygame.display.set_caption("Snake Game")

# Set the color of the snake
snake_color = (169, 27, 96)#A91B60
eaten_color = (255, 0, 128)#FF0080
food_color = (0, 255, 0)
back_color = (0,0,0)
border_color = (255,255,255)

# Set the initial position of the snake
x1 = 100
y1 = 100

# Set the initial velocity of the snake
x1_velocity = 0
y1_velocity = 0

# Set the size of the snake
snake_block = 10
snake_speed = 10
snake_body = [(x1,y1),(x1+10,y1),(x1+20,y1)]

# Set the initial position of the food
foodx = random.randint(1,(SX/10)-1)*10
foody = random.randint(1,(PY/10)-1)*10
#Make sure that food won't spawn at the same position as snake
while foodx == x1 and foody ==y1:
    foodx = random.randint(1,(SX/10)-1)*10
    foody = random.randint(1,(PY/10)-1)*10
    
# Set the initial player score
score = 0

#Set the text box
pygame.font.init() 
font = pygame.font.Font('freesansbold.ttf', 32)
text = font.render('Score: '+str(score), True, (0,0,0), (100,100,100))
textRect = text.get_rect()
textRect.center = (100, 380)

eaten = []

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

    
    
game = True
# Run the game loop
while game==True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game = False
            pygame.quit()
            quit()
            break
        elif event.type == pygame.KEYDOWN:
            if (event.unicode == 'w' or event.key == pygame.K_UP) and y1_velocity != 10:
                x1_velocity = 0
                y1_velocity = -10
            elif (event.unicode == 's' or event.key == pygame.K_DOWN) and y1_velocity !=-10:
                x1_velocity = 0
                y1_velocity = 10
            elif (event.unicode == 'a' or event.key == pygame.K_LEFT) and x1_velocity !=10:
                x1_velocity = -10
                y1_velocity = 0
            elif (event.unicode == 'd' or event.key == pygame.K_RIGHT) and x1_velocity !=-10:
                x1_velocity = 10
                y1_velocity = 0
            elif event.key == pygame.K_ESCAPE:
                game = False
                sleep(0.5)
                pygame.quit()
                quit()
                break
                
    if game == True:
        x1,y1 = snake_body[0]
        neck = snake_body.copy()
        neck.pop(0)

        # Clear the screen
        screen.fill(back_color)

        #Draw text
        screen.blit(text, textRect)

        #Draw borders
        draw_borders(screen, 0, 0, SX, PY+10, 10, border_color)

        # Draw the snake
        #pygame.draw.rect(screen, snake_color, [x1, y1, snake_block, snake_block])
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
        clock.tick(snake_speed)

        # Check for collision with the walls
        if x1 > p_size[0] or x1 <= 0 or y1 >= p_size[1] or y1 <= 0:
            score = -10
            text = font.render('Score: '+str(score), True, (0,0,0), (100,100,100))
            #Reset snake position, reset movement
            x1_velocity = 0
            y1_velocity = 0
            x1 = 100
            y1 = 100
            foodx = random.randint(1,(PX/10)-1)*10
            foody = random.randint(1,(PY/10)-1)*10
            snake_body = [(x1,y1)]

        #Check for collision with body
        if (x1,y1) in neck:
            score = -10
            text = font.render('Score: '+str(score), True, (0,0,0), (100,100,100))
            #Reset snake position, reset movement
            x1_velocity = 0
            y1_velocity = 0
            x1 = 100
            y1 = 100
            foodx = random.randint(1,(PX/10)-1)*10
            foody = random.randint(1,(PY/10)-1)*10
            snake_body = [(x1,y1)]

        # Check for collision with the food
        if x1 == foodx and y1 == foody:
            eaten.append((foodx,foody))
            score += 10
            text = font.render('Score: '+str(score), True, (0,0,0), (100,100,100))
            foodx = random.randint(1,(PX/10)-1)*10
            foody = random.randint(1,(PY/10)-1)*10

        move_snake(snake_body,x1_velocity,y1_velocity)