# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import math
import random

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED=features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

reward=numpy.matrix('2 0 0 0 0 0 0 0 ; 0 2 0 0 0 0 0 0;0 0 2 0 0 0 0 0;0 0 0 2 0 0 0 0;0 0 0 0 2 0 0 0;0 0 0 0 0 2 0 0;0 0 0 0 0 0 2 0 ;0 0 0 0 0 0 0 2')# sparse Reward table
#reward=numpy.matrix('2 1 0 -1 -2 -1 0 1 ; 1 2 1 0 -1 -2 -1 0;0 1 2 1 0 -1 -2 -1;-1 0 1 2 1 0 -1 -2;-2 -1 0 1 2 1 0 -1;-1 -2 -1 0 1 2 1 0;0 -1 -2 -1 0 1 2 1 ;1 0 -1 -2 -1 0 1 2')# detailed Reward table
#reward=numpy.matrix('0 0 0 0 2 0 0 0 ; 0 0 0 0 0 2 0 0;0 0 0 0 0 0 2 0;0 0 0 0 0 0 0 2;2 0 0 0 0 0 0 0;0 2 0 0 0 0 0 0;0 0 2 0 0 0 0 0 ;0 0 0 2 0 0 0 0')# limited Reward table

numpy.set_printoptions(threshold=numpy.nan)
      
class MoveToBeacon(base_agent.BaseAgent):
#"""A Q Learning agent specifically for solving the MoveToBeacon map."""
  seed=1
  alpha=0.1  #learning rate
  gamma=0.3 #discount factor
  epsilon=0.7 #exploration factor
  shift=2 #amount of displacement
  learning_decr=0.0002  # amount of epsilon decreasing each tick
  epsilon_limit=0.1 #minimum value of epsilon

  action_taken=-1
  previous_direction=-1
  Q=numpy.zeros((8,8))  #Q table

 


  def get_direction(self,target,player):    #compute direction of the beacon 
    player_x, player_y= self.player_position(player) 
    if player_x <0 or player_y<0:
      return -1 #error no position available
    del_x =target[0]-player_x
    del_y =target[1]-player_y
    angle= -1*numpy.angle(del_x+1j*del_y, deg=True)  #convert to complex to get angle to beacon, in degrees for ease #opposite angle because of the map orientation [0,0] in top left corner +x to the right +y to the bottom
    if angle<0:
      angle=angle+360.0  #set angle to [0:360[ degrees
    #now return angle as one of 8 directions (R,UR,UC,RL,L,DL,DC,DR,R) as resp {0,1,2,3,4,5,6,7}
    if angle < 22.5 or angle > 337.5:
      return 0 #direction 0 for sure
    direction = math.floor(angle / 22.5) # transform to sextants to avoid hard tests on directions
    if direction %2 ==1: # count two sextants for one octant e.g. UC is 90 deg +-22.5
      direction=direction+1
    
    return math.floor(direction/2)

    
    
  def step(self, obs): #main
 
    global reward
 
    super(MoveToBeacon, self).step(obs)
    random.seed(self.seed)

    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        return actions.FunctionCall(_NO_OP, []) #no beacon, do notihng
      target = [int(neutral_x.mean()), int(neutral_y.mean())]
      
      ######## Q-learning
      compass=self.get_direction(target,player_relative) #get beacon direction
      action_t=0
      if self.action_taken!=-1:
        reward_t=reward[compass,self.action_taken]# get reward
        if compass>-1: #check for valid marine position
          if self.epsilon > self.epsilon_limit: 
            self.epsilon = self.epsilon - self.learning_decr # reduce epsilon each step if above threshold
          action_t=self.pick_action(self.Q,self.epsilon,compass) #choose an action
          #update Q
          self.Q[self.previous_direction,self.action_taken]= self.Q[self.previous_direction,self.action_taken]+self.alpha*(reward_t+self.gamma*max(self.Q[action_t])-self.Q[self.previous_direction,self.action_taken])
          #Compute new position
          target=self.compute_target(action_t,player_relative)
        #else:
         # print("_NO_OP") #little boost to end session : target is the beacon coordinates
 
      self.seed=self.seed+1 #change random seed every step for max randomness
      #save state and action for next step()
      self.previous_direction=compass
      self.action_taken=action_t
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]) #GO !
    else:
      #print("no select")
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
      
  def player_position(self,selected):
    player_y, player_x = (selected == _PLAYER_SELF).nonzero() #get marine position
    if player_y.size<1: #sometimes no position is available
      print("no pos")
      return [-1,-1]
    else:
      return [int(player_x.mean()), int(player_y.mean())]
    
  def pick_action(self,q,e,state):
    action=-1
    #state is beacon direction
    if random.random() <e: 
      action=random.randint(0,7)  #explore mode
    else:
      qq=q[state]
      maxQ=max(qq)   
      action =  [i for i in range(len(qq)) if qq[i] == maxQ] #get index for highest value(s)
      action= random.choice(action) #in case many same MAX values in q[state]
    return action
    
  def compute_target(self,action,player):
    player_x, player_y= self.player_position(player) 

    #print([player_x,player_y])
    if action==0:
      player_x =   player_x +self.shift
      #player_y
    elif action==1:
      player_x =   player_x +self.shift
      player_y =   player_y -self.shift
    elif action==2:
      #player_x =   player_x 
      player_y =   player_y -self.shift
    elif action==3:
      player_x =   player_x -self.shift
      player_y =   player_y -self.shift
    elif action==4:
      player_x =   player_x-self.shift
      #player_y =   player_y 
    elif action==5:
      player_x =   player_x -self.shift
      player_y =   player_y +self.shift
    elif action==6:
      #player_x =   player_x 
      player_y =   player_y +self.shift
    elif action==7:
      player_x =   player_x +self.shift
      player_y =   player_y +self.shift
    else :
      #print("defaulting")
      return [player_x,player_y]
    return [player_x,player_y]
    
  