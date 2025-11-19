import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
from collections import deque
import math


PROCESS_EVERY_N_FRAMES = 2   
KEY_COOLDOWN = 0.18         
GESTURE_THRESHOLD_PIX = 40  
CAM_W, CAM_H = 640, 480     
FPS = 30


PINCH_THRESHOLD_PIX = 40    
HOLD_TIME = 0.5            
RESET_COOLDOWN = 1.0        


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)


class SimpleSnake:
    def __init__(self, grid_size=20, grid_w=24, grid_h=24):
        self.grid_size = grid_size
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.reset()

    def reset(self):
        midx = self.grid_w // 2
        midy = self.grid_h // 2
        self.snake = deque([(midx, midy), (midx-1, midy), (midx-2, midy)])
        self.direction = (1, 0)  
        self.spawn_food()
        self.score = 0
        self.alive = True
        self.step_delay = 0.25    
        self.last_step = time.time()

    def spawn_food(self):
        import random
        while True:
            fx = random.randrange(1, self.grid_w-1)
            fy = random.randrange(1, self.grid_h-1)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                break

    def set_dir(self, d):
        if not self.alive: return
        dx, dy = self.direction
        if d == "up" and (dx,dy) != (0,1):
            self.direction = (0,-1)
        elif d == "down" and (dx,dy) != (0,-1):
            self.direction = (0,1)
        elif d == "left" and (dx,dy) != (1,0):
            self.direction = (-1,0)
        elif d == "right" and (dx,dy) != (-1,0):
            self.direction = (1,0)

    def step(self):
        if not self.alive: return
        now = time.time()
        if now - self.last_step < self.step_delay:
            return
        self.last_step = now
        hx, hy = self.snake[0]
        dx, dy = self.direction
        newh = (hx+dx, hy+dy)
        x,y = newh
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h or newh in self.snake:
            self.alive = False
            return
        self.snake.appendleft(newh)
        if newh == self.food:
            self.score += 1
            self.spawn_food()
            self.step_delay = max(0.05, self.step_delay - 0.005)
        else:
            self.snake.pop()

    def draw(self, surf):
        surf.fill((10,10,10))
        fx, fy = self.food
        pygame.draw.rect(surf, (200,50,50), (fx*self.grid_size, fy*self.grid_size, self.grid_size, self.grid_size))
        for i,(sx,sy) in enumerate(self.snake):
            r = pygame.Rect(sx*self.grid_size, sy*self.grid_size, self.grid_size, self.grid_size)
            color = (0,200,0) if i==0 else (0,120,0)
            pygame.draw.rect(surf, color, r)
        font = pygame.font.SysFont(None, 22)
        surf.blit(font.render(f"Score: {self.score}", True, (255,255,255)), (6,6))
        if not self.alive:
            big = pygame.font.SysFont(None, 40).render("Game Over (R to restart)", True, (255,180,180))
            surf.blit(big, ((surf.get_width()-big.get_width())//2, (surf.get_height()-big.get_height())//2))


pygame.init()
GRID = 20
GAME_W = GRID * 24   # 480
GAME_H = GRID * 24   # 480
screen = pygame.display.set_mode((GAME_W, GAME_H))
pygame.display.set_caption("Snake Game (press R to reset)")
game = SimpleSnake(grid_size=GRID, grid_w=GAME_W//GRID, grid_h=GAME_H//GRID)
clock = pygame.time.Clock()


frame_idx = 0
last_action = None
last_action_time = 0


pinch_start_time = None
last_reset_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h,w,c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = None

    if True:
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mid_up = 8
                mid_down = 5
                x1,y1 = int(hand_landmarks.landmark[mid_up].x*w), int(hand_landmarks.landmark[mid_up].y*h)
                x2,y2 = int(hand_landmarks.landmark[mid_down].x*w), int(hand_landmarks.landmark[mid_down].y*h)

                action = None
                if y1 < y2 and abs(x1-x2) < GESTURE_THRESHOLD_PIX:
                    cv2.putText(frame,"up",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    action = 'up'
                elif y1 > y2 and abs(x1-x2) < GESTURE_THRESHOLD_PIX:
                    cv2.putText(frame,"Down",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    action = 'down'
                elif x1 > x2 and abs(y1-y2) < GESTURE_THRESHOLD_PIX:
                    cv2.putText(frame,"right",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                    action = 'right'
                elif x1 < x2 and abs(y1-y2) < GESTURE_THRESHOLD_PIX:
                    cv2.putText(frame,"left",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
                    action = 'left'
                else:
                    action = None

                if action is not None:
                    now = time.time()
                    if action != last_action or (now - last_action_time) >= KEY_COOLDOWN:
                        game.set_dir(action)
                        last_action = action
                        last_action_time = now

               
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
             
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                dist = math.hypot(tx - ix, ty - iy)

                
                cv2.circle(frame, (tx, ty), 6, (0,255,255), -1)
                cv2.circle(frame, (ix, iy), 6, (0,255,255), -1)

                if dist <= PINCH_THRESHOLD_PIX:
                    cv2.putText(frame, "PINCH", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                    now = time.time()
                    if pinch_start_time is None:
                        pinch_start_time = now
                    if (now - pinch_start_time) >= HOLD_TIME and (now - last_reset_time) >= RESET_COOLDOWN:
                        game.reset()
                        last_reset_time = now
                        pinch_start_time = None
                else:
                    pinch_start_time = None

                break

    cv2.imshow("Gesture Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            break
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_r:
                game.reset()
            elif ev.key == pygame.K_ESCAPE:
                break
            elif ev.key == pygame.K_UP:
                game.set_dir("up")
            elif ev.key == pygame.K_DOWN:
                game.set_dir("down")
            elif ev.key == pygame.K_LEFT:
                game.set_dir("left")
            elif ev.key == pygame.K_RIGHT:
                game.set_dir("right")

    game.step()
    game.draw(screen)
    pygame.display.flip()

    frame_idx += 1
    clock.tick(FPS)

cap.release()
cv2.destroyAllWindows()
pygame.quit()

