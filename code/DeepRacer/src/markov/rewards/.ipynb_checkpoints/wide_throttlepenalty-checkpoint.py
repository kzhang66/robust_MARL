def reward_function(params):
    
    #print('***** NOTEBOOK REWARD FUNCTION - wide centerline ****')
    reward = 1e-3

    distance_from_center = params['distance_from_center']
    track_width = params['track_width']

    # distance from center
    marker_1 = 0.4 * track_width
    
    if distance_from_center <= marker_1:
        reward = 1.0
        
    if params['speed'] < 1.0:
        reward *= 0.5
        
    # to check which reward is being used, let's multiply by -100
    
    #reward *= -100.0

    if params['dist_closed_object'] < 0.8 and params['dist_closed_object'] > 0.5:
        reward *= 0.80
    elif params['dist_closed_object'] < 0.5 and params['dist_closed_object'] > 0.3:
        reward *= 0.50

    return float(reward)