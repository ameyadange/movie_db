import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import json
import re
import argparse
import sys

class MovieRecommendationSystem:
    def __init__(self, credits_file='tmdb_5000_credits.csv', movies_file='tmdb_5000_movies.csv'):
        """
        Initialize the recommendation system
        
        Parameters:
        credits_file: path to credits CSV file
        movies_file: path to movies CSV file
        """
        # Adjustable weights for different features
        self.WEIGHTS = {
            'overview': 0.3,      # Weight for plot/overview similarity
            'genres': 0.25,       # Weight for genre similarity
            'cast': 0.2,          # Weight for actor similarity
            'director': 0.1,      # Weight for director similarity
            'language': 0.05,     # Weight for language similarity
            'popularity': 0.05,   # Weight for popularity score
            'rating': 0.05        # Weight for rating score
        }
        
        self.movies_df = None
        self.credits_df = None
        self.combined_df = None
        self.similarity_matrix = None
        
    def load_data(self, credits_file, movies_file):
        """Load and preprocess the movie data"""
        try:
            print("Loading movie data...")
            self.movies_df = pd.read_csv(movies_file)
            self.credits_df = pd.read_csv(credits_file)
            
            print(f"Movies dataset shape: {self.movies_df.shape}")
            print(f"Credits dataset shape: {self.credits_df.shape}")
            
            # Check if 'id' column exists in both datasets
            if 'id' not in self.movies_df.columns:
                print(f"Available columns in movies dataset: {list(self.movies_df.columns)}")
                if 'movie_id' in self.movies_df.columns:
                    self.movies_df['id'] = self.movies_df['movie_id']
                    print("Using 'movie_id' as 'id' column for movies dataset")
                else:
                    raise ValueError("No 'id' or 'movie_id' column found in movies dataset")
            
            if 'id' not in self.credits_df.columns:
                print(f"Available columns in credits dataset: {list(self.credits_df.columns)}")
                if 'movie_id' in self.credits_df.columns:
                    self.credits_df['id'] = self.credits_df['movie_id']
                    print("Using 'movie_id' as 'id' column for credits dataset")
                else:
                    raise ValueError("No 'id' or 'movie_id' column found in credits dataset")
            
            # Convert id columns to same type
            self.movies_df['id'] = self.movies_df['id'].astype(str)
            self.credits_df['id'] = self.credits_df['id'].astype(str)
            
            # Merge datasets
            print("Merging datasets...")
            self.combined_df = self.movies_df.merge(self.credits_df, on='id', how='inner')
            
            if len(self.combined_df) == 0:
                raise ValueError("No matching IDs found between movies and credits datasets")
            
            print(f"Successfully merged datasets: {len(self.combined_df)} movies")
            
            print("Preprocessing data...")
            self.preprocess_data()
            
            print("Calculating similarity matrix...")
            self.calculate_similarity_matrix()
            
            print(f"Ready! Loaded {len(self.combined_df)} movies successfully!")
            
        except FileNotFoundError as e:
            print(f"Error: Could not find file - {e}")
            print("Please make sure tmdb_5000_credits.csv and tmdb_5000_movies.csv are in the same directory")
            sys.exit(1)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print(f"Error details: {type(e).__name__}")
            if hasattr(e, 'args'):
                print(f"Error args: {e.args}")
            sys.exit(1)
    
    def safe_json_loads(self, text):
        """Safely parse JSON strings, return empty list if invalid"""
        if pd.isna(text):
            return []
        try:
            return json.loads(text.replace("'", '"'))
        except:
            return []
    
    def extract_names(self, json_list, key='name', limit=5):
        """Extract names from JSON list"""
        if not json_list:
            return []
        names = []
        for item in json_list[:limit]:
            if isinstance(item, dict) and key in item:
                names.append(item[key])
        return names
    
    def extract_cast(self, cast_json, limit=5):
        """Extract main cast members"""
        cast_list = self.safe_json_loads(cast_json)
        return self.extract_names(cast_list, 'name', limit)
    
    def extract_director(self, crew_json):
        """Extract director from crew"""
        crew_list = self.safe_json_loads(crew_json)
        for person in crew_list:
            if isinstance(person, dict) and person.get('job') == 'Director':
                return person.get('name', '')
        return ''
    
    def preprocess_data(self):
        """Preprocess the movie data for recommendation"""
        print("Preprocessing movie features...")
        
        # Check for required columns and find alternatives
        required_columns = {
            'title': ['title', 'movie_title', 'name', 'original_title'],
            'overview': ['overview', 'description', 'plot', 'summary'],
            'genres': ['genres', 'genre', 'categories'],
            'cast': ['cast', 'actors', 'starring'],
            'crew': ['crew', 'staff'],
            'popularity': ['popularity', 'score', 'rating_score'],
            'vote_average': ['vote_average', 'rating', 'imdb_score', 'average_rating'],
            'original_language': ['original_language', 'language', 'lang'],
            'release_date': ['release_date', 'year', 'release_year']
        }
        
        # Map columns to their actual names in the dataset
        column_mapping = {}
        for required, alternatives in required_columns.items():
            found = False
            for alt in alternatives:
                if alt in self.combined_df.columns:
                    column_mapping[required] = alt
                    found = True
                    if alt != required:
                        print(f"Using '{alt}' as '{required}' column")
                    break
            if not found:
                print(f"Warning: No column found for '{required}', using defaults")
                column_mapping[required] = None
        
        # Handle missing values and create standardized columns
        if column_mapping['title']:
            self.combined_df['title'] = self.combined_df[column_mapping['title']].fillna('Unknown Title')
        else:
            print("Error: No title column found in dataset")
            print(f"Available columns: {list(self.combined_df.columns)}")
            raise ValueError("Cannot proceed without a title column")
        
        if column_mapping['overview']:
            self.combined_df['overview'] = self.combined_df[column_mapping['overview']].fillna('')
        else:
            print("Warning: No overview column found, using empty strings")
            self.combined_df['overview'] = ''
            
        if column_mapping['genres']:
            self.combined_df['genres'] = self.combined_df[column_mapping['genres']].fillna('[]')
        else:
            print("Warning: No genres column found, creating empty genres")
            self.combined_df['genres'] = '[]'
            
        if column_mapping['cast']:
            self.combined_df['cast'] = self.combined_df[column_mapping['cast']].fillna('[]')
        else:
            print("Warning: No cast column found, creating empty cast")
            self.combined_df['cast'] = '[]'
            
        if column_mapping['crew']:
            self.combined_df['crew'] = self.combined_df[column_mapping['crew']].fillna('[]')
        else:
            print("Warning: No crew column found, creating empty crew")
            self.combined_df['crew'] = '[]'
        
        if column_mapping['popularity']:
            self.combined_df['popularity'] = pd.to_numeric(self.combined_df[column_mapping['popularity']], errors='coerce').fillna(1.0)
        else:
            print("Warning: No popularity column found, using default values")
            self.combined_df['popularity'] = 1.0
            
        if column_mapping['vote_average']:
            self.combined_df['vote_average'] = pd.to_numeric(self.combined_df[column_mapping['vote_average']], errors='coerce').fillna(5.0)
        else:
            print("Warning: No rating column found, using default values") 
            self.combined_df['vote_average'] = 5.0
            
        if column_mapping['original_language']:
            self.combined_df['original_language'] = self.combined_df[column_mapping['original_language']].fillna('en')
        else:
            print("Warning: No language column found, using 'en' as default")
            self.combined_df['original_language'] = 'en'
            
        if column_mapping['release_date']:
            self.combined_df['release_date'] = self.combined_df[column_mapping['release_date']].astype(str).fillna('1900-01-01')
        else:
            print("Warning: No release date column found, using default dates")
            self.combined_df['release_date'] = '1900-01-01'
        
        # Extract features
        print("Extracting genres...")
        self.combined_df['genre_names'] = self.combined_df['genres'].apply(
            lambda x: self.extract_names(self.safe_json_loads(x)) if isinstance(x, str) and x.startswith('[') else [str(x)] if pd.notna(x) else []
        )
        
        print("Extracting cast...")
        self.combined_df['cast_names'] = self.combined_df['cast'].apply(
            lambda x: self.extract_cast(x) if isinstance(x, str) and x.startswith('[') else [str(x)] if pd.notna(x) else []
        )
        
        print("Extracting directors...")
        self.combined_df['director_name'] = self.combined_df['crew'].apply(
            lambda x: self.extract_director(x) if isinstance(x, str) and x.startswith('[') else ''
        )
        
        # Create text features for similarity calculation
        self.combined_df['genre_text'] = self.combined_df['genre_names'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        self.combined_df['cast_text'] = self.combined_df['cast_names'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        self.combined_df['normalized_popularity'] = scaler.fit_transform(
            self.combined_df[['popularity']]
        )
        self.combined_df['normalized_rating'] = scaler.fit_transform(
            self.combined_df[['vote_average']]
        )
        
        # Clean overview text
        self.combined_df['clean_overview'] = self.combined_df['overview'].apply(
            lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()) if pd.notna(x) else ''
        )
        
        print(f"Preprocessing complete. Dataset has {len(self.combined_df)} movies.")
    
    def calculate_similarity_matrix(self):
        """Calculate similarity matrix based on all features"""
        # TF-IDF for text features
        tfidf_overview = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_genres = TfidfVectorizer(max_features=1000)
        tfidf_cast = TfidfVectorizer(max_features=2000)
        
        # Calculate individual similarity matrices
        overview_matrix = cosine_similarity(tfidf_overview.fit_transform(self.combined_df['clean_overview']))
        genre_matrix = cosine_similarity(tfidf_genres.fit_transform(self.combined_df['genre_text']))
        cast_matrix = cosine_similarity(tfidf_cast.fit_transform(self.combined_df['cast_text']))
        
        # Director similarity (exact match)
        directors = self.combined_df['director_name'].values
        director_matrix = np.array([[1 if d1 == d2 and d1 != '' else 0 for d2 in directors] for d1 in directors])
        
        # Language similarity (exact match)
        languages = self.combined_df['original_language'].values
        language_matrix = np.array([[1 if l1 == l2 else 0 for l2 in languages] for l1 in languages])
        
        # Popularity and rating similarity
        popularity_vals = self.combined_df['normalized_popularity'].values
        rating_vals = self.combined_df['normalized_rating'].values
        
        popularity_matrix = 1 - np.abs(popularity_vals[:, np.newaxis] - popularity_vals)
        rating_matrix = 1 - np.abs(rating_vals[:, np.newaxis] - rating_vals)
        
        # Combine all matrices with weights
        self.similarity_matrix = (
            self.WEIGHTS['overview'] * overview_matrix +
            self.WEIGHTS['genres'] * genre_matrix +
            self.WEIGHTS['cast'] * cast_matrix +
            self.WEIGHTS['director'] * director_matrix +
            self.WEIGHTS['language'] * language_matrix +
            self.WEIGHTS['popularity'] * popularity_matrix +
            self.WEIGHTS['rating'] * rating_matrix
        )
    
    def get_movie_recommendations(self, movie_title, num_recommendations=10, verbose=True):
        """Get movie recommendations based on a given movie title"""
        # Make sure we have a title column
        if 'title' not in self.combined_df.columns:
            print("Error: No title column available for search")
            return []
            
        # Find the movie
        movie_matches = self.combined_df[
            self.combined_df['title'].str.contains(movie_title, case=False, na=False)
        ]
        
        if movie_matches.empty:
            if verbose:
                print(f"Movie '{movie_title}' not found in database.")
                similar_titles = self.combined_df[
                    self.combined_df['title'].str.contains(movie_title.split()[0], case=False, na=False)
                ]['title'].head(5).tolist()
                if similar_titles:
                    print("Similar movies found:")
                    for title in similar_titles:
                        print(f"  - {title}")
            return []
        
        # Get the first match
        movie_idx = movie_matches.index[0]
        movie_info = self.combined_df.iloc[movie_idx]
        
        if verbose:
            year = 'N/A'
            if 'release_date' in movie_info and pd.notna(movie_info['release_date']):
                release_date = str(movie_info['release_date'])
                if len(release_date) >= 4:
                    year = release_date[:4]
                    
            print(f"\\nRecommendations for: {movie_info['title']} ({year})")
            
            if 'genre_names' in movie_info and movie_info['genre_names']:
                print(f"Genres: {', '.join(movie_info['genre_names'])}")
            if 'vote_average' in movie_info:
                print(f"Rating: {movie_info['vote_average']}/10")
            print("-" * 80)
        
        # Get similarity scores
        sim_scores = self.similarity_matrix[movie_idx]
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = np.argsort(sim_scores)[::-1][1:num_recommendations+1]
        
        recommendations = []
        
        for i, idx in enumerate(similar_indices, 1):
            rec_movie = self.combined_df.iloc[idx]
            similarity_score = sim_scores[idx]
            
            # Handle year extraction safely
            year = 'N/A'
            if 'release_date' in rec_movie and pd.notna(rec_movie['release_date']):
                release_date = str(rec_movie['release_date'])
                if len(release_date) >= 4:
                    year = release_date[:4]
            
            rec_info = {
                'rank': i,
                'title': rec_movie['title'],
                'year': year,
                'genres': rec_movie.get('genre_names', []),
                'rating': rec_movie.get('vote_average', 0),
                'similarity': similarity_score,
                'overview': rec_movie.get('overview', '')
            }
            
            recommendations.append(rec_info)
            
            if verbose:
                print(f"{i:2d}. {rec_info['title']} ({rec_info['year']})")
                if rec_info['genres']:
                    print(f"     Genres: {', '.join(rec_info['genres'])}")
                print(f"     Rating: {rec_info['rating']}/10 | Similarity: {similarity_score:.3f}")
                if rec_info['overview']:
                    overview = rec_info['overview'][:150] + '...' if len(str(rec_info['overview'])) > 150 else rec_info['overview']
                    print(f"     Overview: {overview}")
                print()
        
        return recommendations
    
    def search_movies(self, search_term, limit=10):
        """Search for movies by title"""
        if 'title' not in self.combined_df.columns:
            print("Error: No title column available for search")
            return []
            
        matches = self.combined_df[
            self.combined_df['title'].str.contains(search_term, case=False, na=False)
        ].head(limit)
        
        results = []
        for _, movie in matches.iterrows():
            # Handle year extraction safely
            year = 'N/A'
            if 'release_date' in movie and pd.notna(movie['release_date']):
                release_date = str(movie['release_date'])
                if len(release_date) >= 4:
                    year = release_date[:4]
                    
            results.append({
                'title': movie['title'],
                'year': year,
                'rating': movie.get('vote_average', 0),
                'genres': movie.get('genre_names', [])
            })
        
        return results
    
    def update_weights(self, **kwargs):
        """Update the weights for different features"""
        for key, value in kwargs.items():
            if key in self.WEIGHTS:
                self.WEIGHTS[key] = float(value)
        
        # Recalculate similarity matrix with new weights
        self.calculate_similarity_matrix()


def main():
    parser = argparse.ArgumentParser(description='TMDB Movie Recommendation System')
    parser.add_argument('movie', nargs='?', help='Movie title to get recommendations for')
    parser.add_argument('-n', '--num', type=int, default=10, help='Number of recommendations (default: 10)')
    parser.add_argument('-s', '--search', action='store_true', help='Search for movies instead of getting recommendations')
    parser.add_argument('--credits', default='tmdb_5000_credits.csv', help='Path to credits CSV file')
    parser.add_argument('--movies', default='tmdb_5000_movies.csv', help='Path to movies CSV file')
    parser.add_argument('-q', '--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--debug', action='store_true', help='Show dataset structure and exit')
    
    # Weight arguments
    parser.add_argument('--overview-weight', type=float, help='Weight for overview similarity (default: 0.3)')
    parser.add_argument('--genre-weight', type=float, help='Weight for genre similarity (default: 0.25)')
    parser.add_argument('--cast-weight', type=float, help='Weight for cast similarity (default: 0.2)')
    parser.add_argument('--director-weight', type=float, help='Weight for director similarity (default: 0.1)')
    parser.add_argument('--language-weight', type=float, help='Weight for language similarity (default: 0.05)')
    parser.add_argument('--popularity-weight', type=float, help='Weight for popularity similarity (default: 0.05)')
    parser.add_argument('--rating-weight', type=float, help='Weight for rating similarity (default: 0.05)')
    
    args = parser.parse_args()
    
    # Debug mode - show dataset structure
    if args.debug:
        try:
            print("=== DATASET DEBUG INFO ===")
            print(f"Loading movies from: {args.movies}")
            movies_df = pd.read_csv(args.movies)
            print(f"Movies dataset shape: {movies_df.shape}")
            print(f"Movies columns: {list(movies_df.columns)}")
            print(f"First few rows of movies:")
            print(movies_df.head(2))
            print()
            
            print(f"Loading credits from: {args.credits}")
            credits_df = pd.read_csv(args.credits)
            print(f"Credits dataset shape: {credits_df.shape}")
            print(f"Credits columns: {list(credits_df.columns)}")
            print(f"First few rows of credits:")
            print(credits_df.head(2))
            
            # Check for common ID columns
            movies_id_cols = [col for col in movies_df.columns if 'id' in col.lower()]
            credits_id_cols = [col for col in credits_df.columns if 'id' in col.lower()]
            print(f"\\nMovies ID-like columns: {movies_id_cols}")
            print(f"Credits ID-like columns: {credits_id_cols}")
            
        except Exception as e:
            print(f"Debug error: {e}")
        return
    
    # Require movie argument if not in debug mode
    if not args.movie:
        print("Error: movie title is required")
        parser.print_help()
        return
    
    # Initialize recommender system
    recommender = MovieRecommendationSystem()
    
    # Update weights if provided
    weight_updates = {}
    if args.overview_weight is not None:
        weight_updates['overview'] = args.overview_weight
    if args.genre_weight is not None:
        weight_updates['genres'] = args.genre_weight
    if args.cast_weight is not None:
        weight_updates['cast'] = args.cast_weight
    if args.director_weight is not None:
        weight_updates['director'] = args.director_weight
    if args.language_weight is not None:
        weight_updates['language'] = args.language_weight
    if args.popularity_weight is not None:
        weight_updates['popularity'] = args.popularity_weight
    if args.rating_weight is not None:
        weight_updates['rating'] = args.rating_weight
    
    if weight_updates and not args.quiet:
        print("Using custom weights:")
        for key, value in weight_updates.items():
            print(f"  {key}: {value}")
        print()
    
    # Load data
    recommender.load_data(args.credits, args.movies)
    
    # Update weights after loading data
    if weight_updates:
        recommender.update_weights(**weight_updates)
    
    # Perform search or recommendations
    if args.search:
        results = recommender.search_movies(args.movie, args.num)
        if results:
            print(f"Movies matching '{args.movie}':")
            print("-" * 60)
            for movie in results:
                print(f"{movie['title']} ({movie['year']}) - {movie['rating']}/10")
                print(f"  Genres: {', '.join(movie['genres'])}")
                print()
        else:
            print(f"No movies found matching '{args.movie}'")
    else:
        recommendations = recommender.get_movie_recommendations(
            args.movie, args.num, verbose=not args.quiet
        )
        
        if not recommendations and not args.quiet:
            # Try searching for similar movies
            search_results = recommender.search_movies(args.movie, 5)
            if search_results:
                print("\\nDid you mean one of these?")
                for movie in search_results:
                    print(f"  - {movie['title']} ({movie['year']})")
        
        if args.quiet and recommendations:
            # Just print movie titles for quiet mode
            for rec in recommendations:
                print(f"{rec['title']} ({rec['year']})")


if __name__ == "__main__":
    main()