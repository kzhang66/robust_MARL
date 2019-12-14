from rl_coach.coach import CoachInterface

coach = CoachInterface(preset='Mujoco_DDPG', level='inverted_pendulum')
coach.run()