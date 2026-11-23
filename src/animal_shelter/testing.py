from animal_shelter.data import load_data
from animal_shelter.features import add_features

animal_outcomes = load_data("./data/test.csv")
with_features = add_features(animal_outcomes)
print(with_features.head())
