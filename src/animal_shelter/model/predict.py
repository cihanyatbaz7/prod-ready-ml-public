from animal_shelter.data import load_data
from animal_shelter.features import add_features
import pickle

def predict(test_data_path: str,model_path: str) -> None:
    test_data = load_data(test_data_path)
    with_features = add_features(test_data)
    
    cat_features = [                                  
        "animal_type",                                        
        "is_dog",                                             
        "has_name",                                           
        "sex",                                                
        "hair_type",                                          
    ]     
                                                        
    num_features = ["days_upon_outcome"]   
    clf_model = load_model(model_path)
    X_test = with_features[cat_features + num_features]
    clf_model.predict(X_test)


def load_model(model_path: str):
    """Load the model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model