{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9080e734-0c11-45e7-b4ac-64d2109375af",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# https://files.grouplens.org/datasets/movielens/ml-25m.zip\n",
    "movies = pd.read_csv(\"movies.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0429b21c-d0e0-4ede-9716-cd3dce0c5bcd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Jumanji (1995)</td>\n",
       "      <td>Adventure|Children|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Grumpier Old Men (1995)</td>\n",
       "      <td>Comedy|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Waiting to Exhale (1995)</td>\n",
       "      <td>Comedy|Drama|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Father of the Bride Part II (1995)</td>\n",
       "      <td>Comedy</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   movieId                               title  \\\n",
       "0        1                    Toy Story (1995)   \n",
       "1        2                      Jumanji (1995)   \n",
       "2        3             Grumpier Old Men (1995)   \n",
       "3        4            Waiting to Exhale (1995)   \n",
       "4        5  Father of the Bride Part II (1995)   \n",
       "\n",
       "                                        genres  \n",
       "0  Adventure|Animation|Children|Comedy|Fantasy  \n",
       "1                   Adventure|Children|Fantasy  \n",
       "2                               Comedy|Romance  \n",
       "3                         Comedy|Drama|Romance  \n",
       "4                                       Comedy  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c559c565-8eda-4858-8006-728a9a1738e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "\n",
    "def clean_title(title):\n",
    "    title = re.sub(\"[^a-zA-Z0-9 ]\", \"\", title)\n",
    "    return title"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "5de95296-51fa-4983-97b7-8719d38bc698",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies[\"clean_title\"] = movies[\"title\"].apply(clean_title)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "73a4cfde-42f0-44f7-b2dc-53dbfbc4c194",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "      <th>clean_title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "      <td>Toy Story 1995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Jumanji (1995)</td>\n",
       "      <td>Adventure|Children|Fantasy</td>\n",
       "      <td>Jumanji 1995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Grumpier Old Men (1995)</td>\n",
       "      <td>Comedy|Romance</td>\n",
       "      <td>Grumpier Old Men 1995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Waiting to Exhale (1995)</td>\n",
       "      <td>Comedy|Drama|Romance</td>\n",
       "      <td>Waiting to Exhale 1995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Father of the Bride Part II (1995)</td>\n",
       "      <td>Comedy</td>\n",
       "      <td>Father of the Bride Part II 1995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62418</th>\n",
       "      <td>209157</td>\n",
       "      <td>We (2018)</td>\n",
       "      <td>Drama</td>\n",
       "      <td>We 2018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62419</th>\n",
       "      <td>209159</td>\n",
       "      <td>Window of the Soul (2001)</td>\n",
       "      <td>Documentary</td>\n",
       "      <td>Window of the Soul 2001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62420</th>\n",
       "      <td>209163</td>\n",
       "      <td>Bad Poems (2018)</td>\n",
       "      <td>Comedy|Drama</td>\n",
       "      <td>Bad Poems 2018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62421</th>\n",
       "      <td>209169</td>\n",
       "      <td>A Girl Thing (2001)</td>\n",
       "      <td>(no genres listed)</td>\n",
       "      <td>A Girl Thing 2001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62422</th>\n",
       "      <td>209171</td>\n",
       "      <td>Women of Devil's Island (1962)</td>\n",
       "      <td>Action|Adventure|Drama</td>\n",
       "      <td>Women of Devils Island 1962</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>62423 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       movieId                               title  \\\n",
       "0            1                    Toy Story (1995)   \n",
       "1            2                      Jumanji (1995)   \n",
       "2            3             Grumpier Old Men (1995)   \n",
       "3            4            Waiting to Exhale (1995)   \n",
       "4            5  Father of the Bride Part II (1995)   \n",
       "...        ...                                 ...   \n",
       "62418   209157                           We (2018)   \n",
       "62419   209159           Window of the Soul (2001)   \n",
       "62420   209163                    Bad Poems (2018)   \n",
       "62421   209169                 A Girl Thing (2001)   \n",
       "62422   209171      Women of Devil's Island (1962)   \n",
       "\n",
       "                                            genres  \\\n",
       "0      Adventure|Animation|Children|Comedy|Fantasy   \n",
       "1                       Adventure|Children|Fantasy   \n",
       "2                                   Comedy|Romance   \n",
       "3                             Comedy|Drama|Romance   \n",
       "4                                           Comedy   \n",
       "...                                            ...   \n",
       "62418                                        Drama   \n",
       "62419                                  Documentary   \n",
       "62420                                 Comedy|Drama   \n",
       "62421                           (no genres listed)   \n",
       "62422                       Action|Adventure|Drama   \n",
       "\n",
       "                            clean_title  \n",
       "0                        Toy Story 1995  \n",
       "1                          Jumanji 1995  \n",
       "2                 Grumpier Old Men 1995  \n",
       "3                Waiting to Exhale 1995  \n",
       "4      Father of the Bride Part II 1995  \n",
       "...                                 ...  \n",
       "62418                           We 2018  \n",
       "62419           Window of the Soul 2001  \n",
       "62420                    Bad Poems 2018  \n",
       "62421                 A Girl Thing 2001  \n",
       "62422       Women of Devils Island 1962  \n",
       "\n",
       "[62423 rows x 4 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "db397894-298f-4a7b-9b8e-790e23138a51",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "vectorizer = TfidfVectorizer(ngram_range=(1,2))\n",
    "\n",
    "tfidf = vectorizer.fit_transform(movies[\"clean_title\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d4b39a12-8bc2-4708-a4d1-8d0d1acbcc00",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "import numpy as np\n",
    "\n",
    "def search(title):\n",
    "    title = clean_title(title)\n",
    "    query_vec = vectorizer.transform([title])\n",
    "    similarity = cosine_similarity(query_vec, tfidf).flatten()\n",
    "    indices = np.argpartition(similarity, -5)[-5:]\n",
    "    results = movies.iloc[indices].iloc[::-1]\n",
    "    \n",
    "    return results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1c85ae9a-5a27-4264-a47c-5816e44e40dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# pip install ipywidgets\n",
    "#jupyter labextension install @jupyter-widgets/jupyterlab-manager"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a515e0f0-5fe8-4bf4-b12d-3a25f8242b22",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "53588c87fb7d4484b7690f27b093e02a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Text(value='Toy Story', description='Movie Title:')"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c8bb023cafba4a9a98e26c6ecddcc10a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Output()"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import ipywidgets as widgets\n",
    "from IPython.display import display\n",
    "\n",
    "movie_input = widgets.Text(\n",
    "    value='Toy Story',\n",
    "    description='Movie Title:',\n",
    "    disabled=False\n",
    ")\n",
    "movie_list = widgets.Output()\n",
    "\n",
    "def on_type(data):\n",
    "    with movie_list:\n",
    "        movie_list.clear_output()\n",
    "        title = data[\"new\"]\n",
    "        if len(title) > 5:\n",
    "            display(search(title))\n",
    "\n",
    "movie_input.observe(on_type, names='value')\n",
    "\n",
    "\n",
    "display(movie_input, movie_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "dacb69cc-1626-4443-82bd-5cd7ceb42db0",
   "metadata": {},
   "outputs": [],
   "source": [
    "movie_id = 89745\n",
    "\n",
    "#def find_similar_movies(movie_id):\n",
    "movie = movies[movies[\"movieId\"] == movie_id]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8bff83ec-c070-4a94-b8a7-097a90590634",
   "metadata": {},
   "outputs": [],
   "source": [
    "ratings = pd.read_csv(\"ratings.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "61bf4ed8-e607-4df8-ab1e-c1eb994d58cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "userId         int64\n",
       "movieId        int64\n",
       "rating       float64\n",
       "timestamp      int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratings.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6c65d587-38a5-4754-a5e8-ec6d6aec3c88",
   "metadata": {},
   "outputs": [],
   "source": [
    "similar_users = ratings[(ratings[\"movieId\"] == movie_id) & (ratings[\"rating\"] > 4)][\"userId\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ea98f79e-b763-4c58-a0f2-c8088859efcf",
   "metadata": {},
   "outputs": [],
   "source": [
    "similar_user_recs = ratings[(ratings[\"userId\"].isin(similar_users)) & (ratings[\"rating\"] > 4)][\"movieId\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "04086b39-a7ef-42f5-9913-469833c2434c",
   "metadata": {},
   "outputs": [],
   "source": [
    "similar_user_recs = similar_user_recs.value_counts() / len(similar_users)\n",
    "\n",
    "similar_user_recs = similar_user_recs[similar_user_recs > .10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e910226b-6f0c-4602-bb82-0696b74899af",
   "metadata": {},
   "outputs": [],
   "source": [
    "all_users = ratings[(ratings[\"movieId\"].isin(similar_user_recs.index)) & (ratings[\"rating\"] > 4)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e520dd69-d82b-4040-b921-2525c81da39a",
   "metadata": {},
   "outputs": [],
   "source": [
    "all_user_recs = all_users[\"movieId\"].value_counts() / len(all_users[\"userId\"].unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4c89e598-8d41-4f16-95a1-f47b7d8af2f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)\n",
    "rec_percentages.columns = [\"similar\", \"all\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "dc0ff4e8-b946-4ce6-8db9-8152e490683d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>similar</th>\n",
       "      <th>all</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.236083</td>\n",
       "      <td>0.126250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>0.103877</td>\n",
       "      <td>0.101516</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>47</th>\n",
       "      <td>0.203115</td>\n",
       "      <td>0.146232</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50</th>\n",
       "      <td>0.211067</td>\n",
       "      <td>0.202959</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>0.182240</td>\n",
       "      <td>0.162835</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>134853</th>\n",
       "      <td>0.198641</td>\n",
       "      <td>0.036444</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>152081</th>\n",
       "      <td>0.133532</td>\n",
       "      <td>0.020652</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>164179</th>\n",
       "      <td>0.128728</td>\n",
       "      <td>0.029124</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>166528</th>\n",
       "      <td>0.124751</td>\n",
       "      <td>0.014411</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>168252</th>\n",
       "      <td>0.132538</td>\n",
       "      <td>0.016469</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>193 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         similar       all\n",
       "1       0.236083  0.126250\n",
       "32      0.103877  0.101516\n",
       "47      0.203115  0.146232\n",
       "50      0.211067  0.202959\n",
       "110     0.182240  0.162835\n",
       "...          ...       ...\n",
       "134853  0.198641  0.036444\n",
       "152081  0.133532  0.020652\n",
       "164179  0.128728  0.029124\n",
       "166528  0.124751  0.014411\n",
       "168252  0.132538  0.016469\n",
       "\n",
       "[193 rows x 2 columns]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rec_percentages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "22666de1-efa7-45b4-a8f1-a66cec82f519",
   "metadata": {},
   "outputs": [],
   "source": [
    "rec_percentages[\"score\"] = rec_percentages[\"similar\"] / rec_percentages[\"all\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "e9b59d85-d8fa-4643-896c-f2dcc24f4d39",
   "metadata": {},
   "outputs": [],
   "source": [
    "rec_percentages = rec_percentages.sort_values(\"score\", ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2d358542-d81d-4d50-b385-57e3baca6d14",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>similar</th>\n",
       "      <th>all</th>\n",
       "      <th>score</th>\n",
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "      <th>clean_title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>17067</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.040459</td>\n",
       "      <td>24.716368</td>\n",
       "      <td>89745</td>\n",
       "      <td>Avengers, The (2012)</td>\n",
       "      <td>Action|Adventure|Sci-Fi|IMAX</td>\n",
       "      <td>Avengers The 2012</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20513</th>\n",
       "      <td>0.103711</td>\n",
       "      <td>0.005289</td>\n",
       "      <td>19.610199</td>\n",
       "      <td>106072</td>\n",
       "      <td>Thor: The Dark World (2013)</td>\n",
       "      <td>Action|Adventure|Fantasy|IMAX</td>\n",
       "      <td>Thor The Dark World 2013</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25058</th>\n",
       "      <td>0.241054</td>\n",
       "      <td>0.012367</td>\n",
       "      <td>19.491770</td>\n",
       "      <td>122892</td>\n",
       "      <td>Avengers: Age of Ultron (2015)</td>\n",
       "      <td>Action|Adventure|Sci-Fi</td>\n",
       "      <td>Avengers Age of Ultron 2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19678</th>\n",
       "      <td>0.216534</td>\n",
       "      <td>0.012119</td>\n",
       "      <td>17.867419</td>\n",
       "      <td>102125</td>\n",
       "      <td>Iron Man 3 (2013)</td>\n",
       "      <td>Action|Sci-Fi|Thriller|IMAX</td>\n",
       "      <td>Iron Man 3 2013</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16725</th>\n",
       "      <td>0.215043</td>\n",
       "      <td>0.012052</td>\n",
       "      <td>17.843074</td>\n",
       "      <td>88140</td>\n",
       "      <td>Captain America: The First Avenger (2011)</td>\n",
       "      <td>Action|Adventure|Sci-Fi|Thriller|War</td>\n",
       "      <td>Captain America The First Avenger 2011</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16312</th>\n",
       "      <td>0.175447</td>\n",
       "      <td>0.010142</td>\n",
       "      <td>17.299824</td>\n",
       "      <td>86332</td>\n",
       "      <td>Thor (2011)</td>\n",
       "      <td>Action|Adventure|Drama|Fantasy|IMAX</td>\n",
       "      <td>Thor 2011</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21348</th>\n",
       "      <td>0.287608</td>\n",
       "      <td>0.016737</td>\n",
       "      <td>17.183667</td>\n",
       "      <td>110102</td>\n",
       "      <td>Captain America: The Winter Soldier (2014)</td>\n",
       "      <td>Action|Adventure|Sci-Fi|IMAX</td>\n",
       "      <td>Captain America The Winter Soldier 2014</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25071</th>\n",
       "      <td>0.214049</td>\n",
       "      <td>0.012856</td>\n",
       "      <td>16.649399</td>\n",
       "      <td>122920</td>\n",
       "      <td>Captain America: Civil War (2016)</td>\n",
       "      <td>Action|Sci-Fi|Thriller</td>\n",
       "      <td>Captain America Civil War 2016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25061</th>\n",
       "      <td>0.136017</td>\n",
       "      <td>0.008573</td>\n",
       "      <td>15.865628</td>\n",
       "      <td>122900</td>\n",
       "      <td>Ant-Man (2015)</td>\n",
       "      <td>Action|Adventure|Sci-Fi</td>\n",
       "      <td>AntMan 2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14628</th>\n",
       "      <td>0.242876</td>\n",
       "      <td>0.015517</td>\n",
       "      <td>15.651921</td>\n",
       "      <td>77561</td>\n",
       "      <td>Iron Man 2 (2010)</td>\n",
       "      <td>Action|Adventure|Sci-Fi|Thriller|IMAX</td>\n",
       "      <td>Iron Man 2 2010</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        similar       all      score  movieId  \\\n",
       "17067  1.000000  0.040459  24.716368    89745   \n",
       "20513  0.103711  0.005289  19.610199   106072   \n",
       "25058  0.241054  0.012367  19.491770   122892   \n",
       "19678  0.216534  0.012119  17.867419   102125   \n",
       "16725  0.215043  0.012052  17.843074    88140   \n",
       "16312  0.175447  0.010142  17.299824    86332   \n",
       "21348  0.287608  0.016737  17.183667   110102   \n",
       "25071  0.214049  0.012856  16.649399   122920   \n",
       "25061  0.136017  0.008573  15.865628   122900   \n",
       "14628  0.242876  0.015517  15.651921    77561   \n",
       "\n",
       "                                            title  \\\n",
       "17067                        Avengers, The (2012)   \n",
       "20513                 Thor: The Dark World (2013)   \n",
       "25058              Avengers: Age of Ultron (2015)   \n",
       "19678                           Iron Man 3 (2013)   \n",
       "16725   Captain America: The First Avenger (2011)   \n",
       "16312                                 Thor (2011)   \n",
       "21348  Captain America: The Winter Soldier (2014)   \n",
       "25071           Captain America: Civil War (2016)   \n",
       "25061                              Ant-Man (2015)   \n",
       "14628                           Iron Man 2 (2010)   \n",
       "\n",
       "                                      genres  \\\n",
       "17067           Action|Adventure|Sci-Fi|IMAX   \n",
       "20513          Action|Adventure|Fantasy|IMAX   \n",
       "25058                Action|Adventure|Sci-Fi   \n",
       "19678            Action|Sci-Fi|Thriller|IMAX   \n",
       "16725   Action|Adventure|Sci-Fi|Thriller|War   \n",
       "16312    Action|Adventure|Drama|Fantasy|IMAX   \n",
       "21348           Action|Adventure|Sci-Fi|IMAX   \n",
       "25071                 Action|Sci-Fi|Thriller   \n",
       "25061                Action|Adventure|Sci-Fi   \n",
       "14628  Action|Adventure|Sci-Fi|Thriller|IMAX   \n",
       "\n",
       "                                   clean_title  \n",
       "17067                        Avengers The 2012  \n",
       "20513                 Thor The Dark World 2013  \n",
       "25058              Avengers Age of Ultron 2015  \n",
       "19678                          Iron Man 3 2013  \n",
       "16725   Captain America The First Avenger 2011  \n",
       "16312                                Thor 2011  \n",
       "21348  Captain America The Winter Soldier 2014  \n",
       "25071           Captain America Civil War 2016  \n",
       "25061                              AntMan 2015  \n",
       "14628                          Iron Man 2 2010  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rec_percentages.head(10).merge(movies, left_index=True, right_on=\"movieId\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "87827a95-5f95-42b8-975e-ff08bcabba88",
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_similar_movies(movie_id):\n",
    "    similar_users = ratings[(ratings[\"movieId\"] == movie_id) & (ratings[\"rating\"] > 4)][\"userId\"].unique()\n",
    "    similar_user_recs = ratings[(ratings[\"userId\"].isin(similar_users)) & (ratings[\"rating\"] > 4)][\"movieId\"]\n",
    "    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)\n",
    "\n",
    "    similar_user_recs = similar_user_recs[similar_user_recs > .10]\n",
    "    all_users = ratings[(ratings[\"movieId\"].isin(similar_user_recs.index)) & (ratings[\"rating\"] > 4)]\n",
    "    all_user_recs = all_users[\"movieId\"].value_counts() / len(all_users[\"userId\"].unique())\n",
    "    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)\n",
    "    rec_percentages.columns = [\"similar\", \"all\"]\n",
    "    \n",
    "    rec_percentages[\"score\"] = rec_percentages[\"similar\"] / rec_percentages[\"all\"]\n",
    "    rec_percentages = rec_percentages.sort_values(\"score\", ascending=False)\n",
    "    return rec_percentages.head(10).merge(movies, left_index=True, right_on=\"movieId\")[[\"score\", \"title\", \"genres\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "556fe392-6a52-4c40-a129-036b4e4435a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "31280088f5d8479d81ce2d3feb2b7db6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Text(value='Toy Story', description='Movie Title:')"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "734504cfca244c5ca4fc0954bd2db04f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Output()"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import ipywidgets as widgets\n",
    "from IPython.display import display\n",
    "\n",
    "movie_name_input = widgets.Text(\n",
    "    value='Toy Story',\n",
    "    description='Movie Title:',\n",
    "    disabled=False\n",
    ")\n",
    "recommendation_list = widgets.Output()\n",
    "\n",
    "def on_type(data):\n",
    "    with recommendation_list:\n",
    "        recommendation_list.clear_output()\n",
    "        title = data[\"new\"]\n",
    "        if len(title) > 5:\n",
    "            results = search(title)\n",
    "            movie_id = results.iloc[0][\"movieId\"]\n",
    "            display(find_similar_movies(movie_id))\n",
    "\n",
    "movie_name_input.observe(on_type, names='value')\n",
    "\n",
    "display(movie_name_input, recommendation_list)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
