import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_mock_interactions(n_users, n_songs, sparsity,save):
    '''
    create mock data for user interactions with songs
    parameters:
        - n_users = number of users 
        - n_songs = number of songs in dataset
        - sparsity = ratio of possible interactions to generate (range of 0,1)
    '''

    total_possible_interactions = n_users * n_songs
    num_interactions = int(total_possible_interactions * sparsity)

    # user power distribution to create more randomness in data
    user_weights = np.random.power(0.3, size=n_users)
    user_weights = user_weights/sum(user_weights)

    song_weights = np.random.exponential(0.3, size=n_songs)
    song_weights = song_weights/sum(song_weights)

    user_ids = np.random.choice(
        np.arrange(n_users),
        size=num_interactions,
        p=user_weights #active users appear more
    )

    song_ids = np.random.choice(
        np.arrange(n_songs),
        size=num_interactions,
        p=song_weights #active users appear more
    )

    user_prefs = np.random.random(n_users)
    interaction_probs = []
    for user_id in user_ids:
        # critical users : more likely to dislike
        if user_prefs[user_id] < 0.2:  
            probs = [0.5, 0.3, 0.2] 
        
        # ethusiastic users: likely to chime
        elif user_prefs[user_id] > 0.8:  
            probs = [0.1, 0.3, 0.6] 

        # original distribution
        else:  
            probs = [0.3, 0.4, 0.3]  
        interaction_probs.append(probs)



    # p.random.randint takes in 3 paramters below: (low, high, size)
    # (lower bound, upper bound, how many numbers to generate)
    # for chime: the interactions will be 0 = dislike, 1 = like, 2 = chime
    interactions = pd.DataFrame({
        "user_id": user_ids,
        "song_id" : song_ids,
        "interaction": [np.random.choice([0,1,2], num_interactions,p=probs) for probs in interaction_probs],
        "listen_count":np.random.choice(range(1,6), num_interactions, p=[0.5, 0.25, 0.15, 0.07, 0.03]),
        "timestamp": [
            datetime.now() - timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            ) for _ in range(num_interactions)
        ]
    })

    #removes duplicates for user-song combinations (same user can't rate the same song more than once)
    interactions = interactions.drop_duplicates(subset=['user_id', 'song_id'])

    if save:
        path = Path('training_data/data.csv')
        interactions.to_csv(path, index=False)


    #data
    return interactions.sort_values(['user_id','timestamp'])

#mock_data
interactions_df = create_mock_interactions(100,500,0.1, True)

#generate user-item matrix
user_item_matrix = interactions_df.pivot(
    index='user_id', 
    columns='song_id', 
    values='interaction'
).fillna(0)

print(interactions_df.head())


