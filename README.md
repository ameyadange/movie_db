# Movie Recommender App

This is a CLI-based movie recommendation app that uses content-based filtering to recommend movies based on the input. 
It uses the TMDB database with 5000 movies (https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download&select=tmdb_5000_movies.csv)


# Input
movie_db % python3 movie_recommender.py "The Godfather"

# Output
Loading movie data...\
Movies dataset shape: (4803, 20)\
Credits dataset shape: (4803, 4)\
Available columns in credits dataset: ['movie_id', 'title', 'cast', 'crew']\
Using 'movie_id' as 'id' column for credits dataset\
Merging datasets...\
Successfully merged datasets: 4803 movies\
Preprocessing data...\
Preprocessing movie features...\
Using 'original_title' as 'title' column\
Extracting genres...\
Extracting cast...\
Extracting directors...\
Preprocessing complete. Dataset has 4803 movies.\
Calculating similarity matrix...\
Ready! Loaded 4803 movies successfully!\
\nRecommendations for: The Godfather: Part III (1990)\
Genres: Crime, Drama, Thriller\
Rating: 7.1/10

 1. The Godfather: Part II (1974)
     Genres: Drama, Crime
     Rating: 8.3/10 | Similarity: 0.554
     Overview: In the continuing saga of the Corleone crime family, a young Vito Corleone grows up in Sicily and in 1910s New York. In the 1950s, Michael Corleone at...

 2. The Godfather (1972)
     Genres: Drama, Crime
     Rating: 8.4/10 | Similarity: 0.502
     Overview: Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corl...

 3. The Outsiders (1983)
     Genres: Crime, Drama
     Rating: 6.9/10 | Similarity: 0.453
     Overview: When two poor greasers, Johnny, and Ponyboy are assaulted by a vicious gang, the socs, and Johnny kills one of the attackers, tension begins to mount ...

 4. We Own the Night (2007)
     Genres: Drama, Crime, Thriller
     Rating: 6.5/10 | Similarity: 0.441
     Overview: A New York nightclub manager tries to save his brother and father from Russian mafia hitmen.

 5. A Most Violent Year (2014)
     Genres: Crime, Drama, Thriller
     Rating: 6.5/10 | Similarity: 0.429
     Overview: A thriller set in New York City during the winter of 1981, statistically one of the most violent years in the city's history, and centered on a the li...

 6. Donnie Brasco (1997)
     Genres: Crime, Drama, Thriller
     Rating: 7.4/10 | Similarity: 0.426
     Overview: An FBI undercover agent infilitrates the mob and finds himself identifying more with the mafia life at the expense of his regular one.

 7. The Talented Mr. Ripley (1999)
     Genres: Thriller, Crime, Drama
     Rating: 7.0/10 | Similarity: 0.418
     Overview: Tom Ripley is a calculating young man who believes it's better to be a fake somebody than a real nobody. Opportunity knocks in the form of a wealthy U...

 8. The Lincoln Lawyer (2011)
     Genres: Crime, Drama, Thriller
     Rating: 7.0/10 | Similarity: 0.415
     Overview: A lawyer conducts business from the back of his Lincoln town car while representing a high-profile client in Beverly Hills.

 9. Sexy Beast (2000)
     Genres: Crime, Drama, Thriller
     Rating: 7.0/10 | Similarity: 0.415
     Overview: Gary is a former gangster who has made a modest amount of money from his criminal career. Happy to put his life of crime behind him, he has retired wi...

10. American Psycho (2000)
     Genres: Thriller, Drama, Crime
     Rating: 7.3/10 | Similarity: 0.412
     Overview: A wealthy New York investment banking executive hides his alternate psychopathic ego from his co-workers and friends as he escalates deeper into his i...