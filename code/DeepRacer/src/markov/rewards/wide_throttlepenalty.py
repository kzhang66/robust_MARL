def reward_function(params):
   
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    dist_closest_bot_car = params['dist_closest_bot_car']

    # distance from center
    marker_1 = 0.4 * track_width
    
    reward = 1e-3  # likely crashed / close to off track

    if distance_from_center <= marker_1:
        reward = 1.0
        
    if params['speed'] < 1.0:
        reward *= 0.5

    if 0.5 <= dist_closest_bot_car < 0.8:
        reward *= 0.80
    elif 0.3 < dist_closest_bot_car < 0.5:
        reward *= 0.50

    return float(reward)