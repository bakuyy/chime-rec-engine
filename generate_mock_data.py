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

    # p.random.randint takes in 3 paramters below: (low, high, size)
    # (lower bound, upper bound, how many numbers to generate)
    # for chime: the interactions will be 0 = dislike, 1 = like, 2 = chime
    interactions = pd.DataFrame({
        "user_id": np.random.randint(0, n_users, num_interactions),
        "song_id" : np.random.randint(0, n_songs, num_interactions),
        "interaction": np.random.choice([0,1,2], num_interactions,p=[0.3,0.4,0.3]),
        "listen_count":np.random.choice(range(1,6), num_interactions, p=np.linspace(0.5, 0.1, 5)/sum(np.linspace(0.5, 0.1, 5))),
        "timestamp":[datetime.now() - timedelta(days=np.random.randint(0, 365)) 
                     for _ in range(num_interactions)]
    })

    #removes duplicates for user-song combinations (same user can't rate the same song more than once)
    interactions = interactions.drop_duplicates(subset=['user_id', 'song_id'])

    if save:
        path = Path('training_data/data.csv')
        interactions.to_csv(path, index=False)


    #data
    return interactions.sort_values(['user_id','timestamp'])

#mock_data
interactions_df = create_mock_interactions(50,200,0.1, True)

#generate user-item matrix
user_item_matrix = interactions_df.pivot(
    index='user_id', 
    columns='song_id', 
    values='interaction'
).fillna(0)

print(interactions_df.head())


