OBSERVATION_SPACE_LIMIT = {
    'T_room': (5, 60),
    'T_wall': (5, 60),
    'T_hp_ret': (5, 65),
    'T_hp_sup': (5, 65),
    'T_amb': (-25, 45),
    'T_forecast' : (-25, 45),
    'Qdot_gains': (0, 8000),
    'goal_temperature': (15, 30),  # Goal temperature range for goal-based learning
}
