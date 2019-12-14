""" Instead of the bubble penalty, we consider both the  
distance between the learner car and the closest bot car,  
as well as if the learner car is within the wedge area 
apexed at the bot car. The assumption is within the wedge
is more possible to crash. """

import math

def calc_angle(x1, y1, x2, y2):
    return math.atan2(y2-y1, x2-x1)

def learner_behind_bot(params):
    """ return True if the learner car is behind the closest bot car, otherwise False """
    
    SAFE_ANGLE = 30 # 45
    
    bot_heading = params['bot_heading']
    
    # angle from the learner car to the closest bot car
    angle_to_bot = calc_angle(params['x'], params['y'], params['bot_x'], params['bot_y'])
    
    # compute the angle between the heading of the bot car and the angle_to_bot
    # there are four scenarios discussed in the Quip doc: 
    abs_diff = abs(angle_to_bot - bot_heading)
    if params['closest_waypoints'][1] <= params['bot_closest_waypoints'][1]:
        # learner car behind bot car
        if angle_to_bot * bot_heading >= 0:
            angle_to_heading = abs_diff
        else:
            angle_to_heading = abs_diff if abs_diff  < 90 else 180 - abs_diff
    else:
        # learner car ahead of bot car
        if angle_to_bot * bot_heading >= 0:
            angle_to_heading = 180 - abs_diff
        else:
            angle_to_heading = 180 - abs_diff if abs_diff < 90 else abs_diff
            
    if angle_to_heading < SAFE_ANGLE or angle_to_heading > 180 - SAFE_ANGLE:
        flag_behind = True
    else:
        flag_behind = False    
        
    return flag_behind
    

def reward_function(params):

    # learner car
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    heading = params['heading']
    steering_angle = params['steering_angle']
    waypoints = params['waypoints']

    dist_closest_bot_car = params['dist_closest_bot']

    reward = 1e-3  # likely crashed / close to off track

    # wide centerline
    marker_1 = 0.4 * track_width
    if distance_from_center <= marker_1:
        reward = 1.0

    # speed penalty
    if params['speed'] < 1.0:
        reward *= 0.5
        
    flag_behind = learner_behind_bot(params)

    # penalize if distance too close
    if 0.5 <= dist_closest_bot_car < 0.8 and flag_behind:
        reward *= 0.80
    elif 0.3 < dist_closest_bot_car < 0.5 and flag_behind:
        reward *= 0.50        


    return float(reward)