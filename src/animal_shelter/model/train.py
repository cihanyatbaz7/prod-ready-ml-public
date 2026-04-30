from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle


from animal_shelter.data import load_data
from animal_shelter.features import add_features


def train(input_path: str, output_path: str) -> None:
    """Train the model and save it to disk."""

    print("Training the model...")
    raw_data = load_data(input_path)
    data_with_features = add_features(raw_data)
    
    cat_features = [                                  
        "animal_type",                                        
        "is_dog",                                             
        "has_name",                                           
        "sex",                                                
        "hair_type",                                          
    ]                                                         
    num_features = ["days_upon_outcome"]                  

    clf_model = construct_pipeline(num_features, cat_features)
                                                            
    X = data_with_features[cat_features + num_features]
    y = data_with_features["outcome_type"]

    trained_model = clf_model.fit(X, y)

    # Save the model to disk.
    with open(output_path, "wb") as f:
        pickle.dump(trained_model, f)   
        print(f"Model saved to {output_path}")

def construct_pipeline(num_features: list, cat_features: list) -> Pipeline:
    """Construct the training pipeline."""
    num_transformer = Pipeline(                                                
        steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]     
    )                                                                          
    cat_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])

    transformer = ColumnTransformer(                                           
        (                                                                      
            ("numeric", num_transformer, num_features),                        
            ("categorical", cat_transformer, cat_features),                    
        )                                                                      
    )

    clf_model = Pipeline(                                                      
        [("transformer", transformer), ("model", RandomForestClassifier())]    
    )
    return clf_model    