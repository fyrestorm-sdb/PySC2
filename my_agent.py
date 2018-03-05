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


seed=1
alpha=0.1  #learning rate
gamma=0.3 #discount factor
epsilon=0.7 #exploration factor
shift=2 #amount of displacement



Q=numpy.zeros((8,8))  #Q table
reward=numpy.matrix('2 1 0 0 -1 -2 1 -1 ; 1 2 1 -1 -2 -1 0 0;0 1 2 -2 -1 0 -1 1;0 -1 -2 2 1 0 1 -1;-1 -2 -1 1 2 1 0 0;-2 -1 0 0 1 2 -1 1;1 0 -1 1 0 -1 2 -2;-1 0 1 -1 0 1 -2 2')# Reward table

numpy.set_printoptions(threshold=numpy.nan)
      
class MoveToBeacon(base_agent.BaseAgent):
#"""A Q Learning agent specifically for solving the MoveToBeacon map."""




  def get_direction(self,target,player):    #retriive direction of the beacon 
    #print(target)
    player_x, player_y= self.player_position(player) 
    if player_x <0 or player_y<0:
      return -1
    del_x =target[0]-player_x
    del_y =target[1]-player_y
    #print(del_x, del_y )
    angle= -1*numpy.angle(del_x+1j*del_y, deg=True)  #convert to complex to get angle to beacon, in degrees for ease #opposite angle because of the map orientation [0,0] in top left corner +x to the right +y to the bottom
    if angle<0:
      angle=angle+360.0  #set angle to [0:360[ degrees
    #print(angle )
    #now return angle as one of 8 directions (R,UR,UC,RL,L,DL,DC,DR,R) as resp {0,1,2,3,4,5,6,7}
    if angle < 22.5 or angle > 337.5:
      return 0 #direction 0 for sure
    direction = math.floor(angle / 22.5) # transform to sextants to avoid hard tests on directions
    if direction %2 ==1: # count two sextants for one octant e.g. UC is 90 deg +-22.5
      direction=direction+1
    
    direction=math.floor(direction/2)
    return direction
    #print(direction)
    #print("----------")
    
    
    
  def step(self, obs): #main
    global Q
    global seed
    super(MoveToBeacon, self).step(obs)
    random.seed(seed)
    #print(obs.observation["available_actions"])
    #print("------------------")
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      selected=obs.observation["screen"][_PLAYER_RELATIVE] 
      #player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
      #print((player_relative == player_relative).nonzero())
    
      #print("------------------")
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        return actions.FunctionCall(_NO_OP, [])
      #player_y, player_x= self.player_position(selected) 
      target = [int(neutral_x.mean()), int(neutral_y.mean())]
      compass=self.get_direction(target,selected)
      #action=self.pick_action(Q,epsilon,compass) #choose an action
      print("beacon:")
      print(target)
      print("compass:")
      print(compass)
      if compass>-1:
        target=self.compute_target(compass,selected)
      else:
        print("_NO_OP") #little boost to end session
        #return actions.FunctionCall(_NO_OP, [])      
      print("target:")
      print(target)
      print("----------")
      seed=seed+1 #change random seed every step
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
    else:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
      
  def player_position(self,selected):
   # selected=obs.observation["screen"][_SELECTED]
    player_y, player_x = (selected == _PLAYER_SELF).nonzero() #get marine position
    if player_y.size<1: #sometimes no position is available
      print("no pos")
      return [-1,-1]
    else:
      return [int(player_x.mean()), int(player_y.mean())]
    
  def pick_action(self,q,e,state):
    action=-1
    #state is beacon direction
    if random.random <e: 
      action=random.randint(0,7)  #explore mode
    else:
      maxQ=max(q[state])
      action =  [i for i in range(len(q[state])) if q[i] == maxQ]
      action= random.choice(action) #in case many same MAX values in Q
    return action
    
  def compute_target(self,action,player):
    player_x, player_y= self.player_position(player) 
    #print("len(player_y):")
    #print(player_y.size)  
  # player_x = player_x
    #player_y = player_y[0]
    
    print("player:")
    print([player_x,player_y])
    if action == 0:
      player_x =   player_x +shift
      #player_y
    elif action ==1:
      player_x =   player_x +shift
      player_y =   player_y -shift
    elif action==2:
      #player_x =   player_x 
      player_y =   player_y -shift
    elif action==3:
      player_x =   player_x -shift
      player_y =   player_y -shift
    elif action==4:
      player_x =   player_x-shift
      #player_y =   player_y 
    elif action ==5:
      player_x =   player_x -shift
      player_y =   player_y +shift
    elif action==6:
      #player_x =   player_x 
      player_y =   player_y +shift
    elif action==7:
      player_x =   player_x +shift
      player_y =   player_y +shift
    else :
      print("defauting")
      return [player_x,player_y]
    return [player_x,player_y]