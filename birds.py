def better_sort(movies):
    shifted_movies = []
    unshifted_movies = []
    for movie in movies:
        if movie.lower().startswith("The "):
            shifted_movies.append(movie[4:] + ", The")
        else:
            unshifted_movies.append(movie)
            shifted_movies.sort()
            unshifted_movies.sort()
    return unshifted_movies + shifted_movies
        
# driver cases -- ignore
movies = ["Avatar", "Casper", "The Aviator", "The Birds", "Zoolander"]
print(better_sort(movies))

movies = ["Black Panther", "Avatar: The Way of Water", "Fantastic Beasts and Where to Find Them", "Jumanji: The Next Leve;"
          "The Chronicles of Narnia: The Lion, The Witch and The Wardrobe", "The Hobbit: the Battle of the Five Armies", "Thelma and Louise"]
print(better_sort(movies))