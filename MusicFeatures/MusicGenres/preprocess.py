import numpy as np
import pandas as pd
from collections import Counter
import torch as tr

raw_genre_path = "fma_metadata/raw_genres.csv"
raw_tracks_path = "fma_metadata/raw_tracks.csv"
genre_path = "fma_metadata/genres.csv"
tracks_path = "fma_metadata/tracks.csv"

raw_genre_path_pd = pd.read_csv(raw_genre_path)
raw_tracks_path_pd = pd.read_csv(raw_tracks_path, low_memory=False)
genre_path_pd = pd.read_csv(genre_path)
tracks_path_pd = pd.read_csv(tracks_path, low_memory=False)


# get the root genre
def get_parent(child_node):
    if len(genre_path_pd['parent'][genre_path_pd['genre_id'] == child_node]) > 0:
        parent = genre_path_pd['parent'][genre_path_pd['genre_id'] == child_node].item()
        if parent == 0:
            return child_node
        else:
            while parent != 0:
                child_node = genre_path_pd.loc[genre_path_pd['genre_id'] == parent]['genre_id'].item()
                parent = genre_path_pd['parent'][genre_path_pd['genre_id'] == child_node].item()
            return child_node
    else:
        return child_node


root_genre_dict = {}

# convert a genre to its root genre
for item in genre_path_pd.iterrows():
    child_node = item[1][0]
    root_genre = get_parent(child_node)
    root_genre_dict[child_node] = root_genre

num_top_genres = 8
genre_count = dict()
idx_and_label = dict()

# create the data dictionary whose key is the track id and value is the genre
for idx, track_genres in enumerate(raw_tracks_path_pd['track_genres'].tolist()):
    if track_genres == track_genres:
        track_genres = eval(track_genres)
        # a song might has multiple class, so I use a set to store genres of a song.
        genres = set()
        for genre in track_genres:
            genre_id = int(genre['genre_id'])
            parent = root_genre_dict[genre_id]
            genre_count[parent] = genre_count.get(parent, 0) + 1
            genres.add(parent)
        idx_and_label[raw_tracks_path_pd.iloc[idx]['track_id']] = genres

genre_counter = Counter(genre_count)
# get top 10 genres
top_10_genres = genre_counter.most_common(num_top_genres)
# get list of index of top 10 genres
top_10_genres_list = [item[0] for item in top_10_genres]
# get list of name corresponding to the top 10 genres
top_10_genres_string = [genre_path_pd['title'][genre_path_pd['genre_id'] == idx].item() for idx in top_10_genres_list]

# list of top 10 data
top_10_genres_data = []
for item in idx_and_label.items():
    interset = set(top_10_genres_list).intersection(item[1])
    if len(interset) > 0:
        top_10_genres_data.append((item[0], interset))

# index to genres {genres id: genres title}
raw_idx2genres = dict(zip(top_10_genres_list, top_10_genres_string))
# genres to index {genres title: index}
idx2genres = dict(zip(np.arange(10), top_10_genres_string))
# index to genres {index: genres title}
rawidx2newidx = dict(zip(top_10_genres_list, np.arange(num_top_genres)))

# convert labels to one hot labels
top_10_onehot_data = list()
for item in top_10_genres_data:
    label = [rawidx2newidx[raw_label] for raw_label in item[1]]
    label = tr.sum(tr.nn.functional.one_hot(tr.tensor(label),num_classes=num_top_genres),axis=0).tolist()
    top_10_onehot_data.append((item[0],label))

# save the preprocessed data
np.save("processed_data/top_10_onehot_data", top_10_onehot_data)
np.save("processed_data/idx_dict_list",{"raw_idx2genres":raw_idx2genres,"idx2genres":idx2genres,"rawidx2newidx":rawidx2newidx})
