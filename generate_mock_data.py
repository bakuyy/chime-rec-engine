import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from random import sample
from collections import defaultdict
import random
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
    song_weights = np.random.exponential(0.3, size=n_songs)
    song_weights = song_weights/sum(song_weights)

    # make some songs unplayed
    zero_mask = np.random.random(n_songs)<0.2
    song_weights[zero_mask] = 0

    song_weights = song_weights/sum(song_weights)

    user_ids = np.random.randint(0,n_users,num_interactions)
    song_ids = np.random.choice(
        np.arange(n_songs),
        size=num_interactions,
        p=song_weights #active users appear more
    )

    genres = [
    "Rock", "Pop", "Hip-Hop", "Jazz", "Classical", "Electronic", "Country", "R&B", "Reggae", "Blues", 
    "Funk", "Soul", "Gospel", "Metal", "Punk", "Alternative", "Folk", "Latin", "Disco", "K-Pop", 
    "House", "Techno", "Trap", "Ambient", "World", "Ska", "Indie", "Grunge", "Opera", "Dubstep", 
    "Lo-fi", "Afrobeats", "Synthwave", "Chillstep", "DrumAndBass", "EDM", "Dancehall", "Trance", "Emo", "Acoustic"
    ]

    hashtags = [
    "#InMyFeelings", "#TikTokViral", "#SadBoyHours", "#Heartbreak", "#Mood", "#Emotional", "#Relatable", "#LateNightVibes", "#Soulful", "#LoveSong", 
    "#DeepThoughts", "#ChillVibes", "#BrokenHeart", "#CryingInTheClub", "#LostInThought", "#MusicToFeelTo", "#SelfReflection", "#LifeFeels", "#Dreamy", 
    "#OnMyMind", "#FeelingsMatter", "#MusicMood", "#StoryOfMyLife", "#RawEmotions", "#ViralSound", "#LostAndFound", "#CurrentMood", "#InnerPeace", 
    "#HealingThroughMusic", "#FeelsTrain", "#JustVibing", "#AloneButNotLonely", "#MusicTherapy"
    ]

    song_genres = {}
    song_hashtags = {}
    for song_id in range(n_songs):
        num_genres = np.random.randint(0,len(genres)//2)
        num_hashtags = np.random.randint(0,len(hashtags)//2)

        song_genres[song_id]=random.sample(genres, num_genres)
        song_hashtags[song_id] = np.random.choice(hashtags,num_hashtags, replace=False)

    # p.random.randint takes in 3 paramters below: (low, high, size)
    # (lower bound, upper bound, how many numbers to generate)
    # for chime: the interactions will be 0 = dislike, 1 = like, 2 = chime
    #removes duplicates for user-song combinations (same user can't rate the same song more than once)
    interactions = pd.DataFrame({
        "user_id": user_ids,
        "song_id": song_ids,
        "listen_count":np.random.choice(range(0,5), num_interactions, p=[0.2, 0.4, 0.3, 0.07, 0.03]),
        "timestamp": [
            datetime.now() - timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            ) for _ in range(num_interactions)
        ]
    }).drop_duplicates(subset=["user_id", "song_id"])

    user_prefs = np.random.random(n_users)
    def assign_interaction(user_id):
        if user_prefs[user_id] < 0.4:  # critical users: more likely to dislike
            return np.random.choice([-1, 1, 2], p=[0.7, 0.2, 0.1])
        elif user_prefs[user_id] > 0.8:  # enthusiastic users: more likely to chime
            return np.random.choice([-1, 1, 2], p=[0.2, 0.5, 0.3])
        else:  # neutral users
            return np.random.choice([-1, 1, 2], p=[0.5, 0.3, 0.2])

    interactions["interaction"] = interactions["user_id"].apply(assign_interaction)
    interactions["genres"] = interactions["song_id"].apply(lambda x: list(song_genres[x]))
    interactions["hashtag"] = interactions["song_id"].apply(lambda x: ",".join(song_hashtags[x]))

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
